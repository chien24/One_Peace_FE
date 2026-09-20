"""Trích xuất đặc trưng VISUAL cho UniAV bằng ONE-PEACE video encoder (fine-tune K400).

Tái tạo cách tác giả UniAV làm (README + mục Implementation Details của bài báo):
  video -> 16 fps -> resize cạnh ngắn 256 -> center crop 256x256
        -> cửa sổ 16 frame, bước `--stride` frame
        -> ONE-PEACE video backbone -> trung bình CLS token 16 frame -> 1536-d.
Kết quả: <output_dir>/<video_id>_one_peace_video_finetune.npy, shape (T, 1536), float32.

  stride 4 (0.25 s): DESED, UnAV-100      stride 8 (0.5 s): ActivityNet-1.3

Ví dụ (Colab):
  python extract_video_features.py --checkpoint onepeace_video_k400.pth \\
      --video_dir /content/YouCookII/videos --output_dir /content/feats/youcookii \\
      --ids_from annotations/youcookii_all.json --stride 8
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from collections.abc import Callable

import numpy as np
import torch

from onepeace_video_backbone import IMG_MEAN, IMG_STD, OnePeaceViT, load_k400_checkpoint
from video_io import (
    find_ffmpeg,
    iter_batches,
    iter_clips,
    iter_video_frames,
    list_video_ids,
    num_windows,
    prefetch,
    probe,
    save_npy_atomic,
    shard,
)

DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, help="onepeace_video_k400.pth")
    p.add_argument("--video_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--ids_from", default=None, help="json annotation / txt danh sách id (mặc định: mọi .mp4)")
    p.add_argument("--video_ext", default=".mp4")
    p.add_argument("--fps", type=int, default=16)
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--num_frames", type=int, default=16)
    p.add_argument("--stride", type=int, default=8, help="bước cửa sổ (frame). 4 = 0.25 s, 8 = 0.5 s")
    p.add_argument("--batch_size", type=int, default=8, help="số clip / lần forward")
    p.add_argument(
        "--dtype",
        choices=["auto", *DTYPES],
        default="auto",
        help="auto = fp16 trên GPU, fp32 trên CPU. fp32 sát bản gốc nhất nhưng chậm hơn nhiều lần; "
        "bf16 chỉ dùng khi fp16 báo NaN/inf",
    )
    p.add_argument(
        "--resize_backend",
        choices=["cv2", "ffmpeg"],
        default="cv2",
        help="cv2 = resize/crop y hệt mmaction2 của ONE-PEACE (mặc định); ffmpeg = nhanh hơn",
    )
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--no_sdpa", action="store_true", help="dùng attention matmul gốc thay vì SDPA")
    p.add_argument(
        "--compile",
        action="store_true",
        help="(GPU) torch.compile: nhanh hơn ~7%% trên RTX 3060, nhiều hơn trên A100; mất ~1 phút "
        "biên dịch lúc đầu. Linux/Colab dùng được ngay; Windows cần `pip install triton-windows`",
    )
    p.add_argument("--num_shards", type=int, default=1, help="chia danh sách video cho nhiều phiên chạy")
    p.add_argument("--shard_id", type=int, default=0)
    p.add_argument("--limit", type=int, default=None, help="chỉ xử lý N video đầu (để thử)")
    p.add_argument("--max_clips", type=int, default=None, help="chỉ lấy N clip đầu mỗi video (để thử)")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--log_every", type=float, default=60, help="in tiến độ trong video mỗi N giây")
    return p.parse_args()


def resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    if device.type == "cpu":
        if name not in ("auto", "fp32"):
            print("[warn] CPU: chuyển sang fp32")
        return torch.float32
    return torch.float16 if name == "auto" else DTYPES[name]


def build_encoder(
    model: OnePeaceViT, device: torch.device, dtype: torch.dtype, batch_size: int, compile_model: bool
) -> Callable[[np.ndarray], torch.Tensor]:
    """Hàm (B, T, H, W, 3) uint8 -> (B, 1536) float32 trên ``device`` (chưa đồng bộ về CPU)."""
    mean = torch.tensor(IMG_MEAN, device=device).view(1, 3, 1, 1, 1)
    std = torch.tensor(IMG_STD, device=device).view(1, 3, 1, 1, 1)
    pin = device.type == "cuda"
    forward = model.extract_clip_features
    if compile_model:
        forward = torch.compile(forward, dynamic=False)

    @torch.inference_mode()
    def encode(clips: np.ndarray) -> torch.Tensor:
        n = len(clips)
        if compile_model and n < batch_size:
            # batch cuối của video: lặp clip cuối cho đủ batch, tránh biên dịch lại cho shape mới
            clips = np.concatenate([clips, np.repeat(clips[-1:], batch_size - n, axis=0)])
        x = torch.from_numpy(clips)
        if pin:
            x = x.pin_memory()
        x = x.to(device, non_blocking=True).permute(0, 4, 1, 2, 3).float()  # B 3 T H W
        x = ((x - mean) / std).to(dtype)
        return forward(x)[:n].float()

    return encode


def extract_one(
    path: str, encode: Callable[[np.ndarray], torch.Tensor], args: argparse.Namespace, ffmpeg: str, tag: str
) -> np.ndarray:
    """Đặc trưng (T, 1536) của một video."""
    t_start = time.time()
    info = probe(path, ffmpeg)
    duration = info["duration"] or 0.0
    n_expected = num_windows(int(duration * args.fps), args.num_frames, args.stride)
    if args.max_clips is not None:
        n_expected = min(n_expected, args.max_clips)
    print(f"{tag}: {duration:.0f}s ~ {n_expected} clip", flush=True)

    frames = iter_video_frames(path, args.fps, args.size, ffmpeg, args.resize_backend, info=info)
    batches = iter_batches(iter_clips(frames, args.num_frames, args.stride), args.batch_size, args.max_clips)

    feats, done, t_log = [], 0, time.time()
    # giải mã + resize ở thread nền, GPU không phải chờ CPU
    for batch in prefetch(batches, max_items=3):
        feats.append(encode(batch))  # giữ trên GPU, chỉ đồng bộ một lần khi xong video
        done += len(batch)
        if time.time() - t_log > args.log_every:
            rate = done / (time.time() - t_start)
            eta = max(n_expected - done, 0) / max(rate, 1e-9)
            print(f"    {done}/{n_expected} clip | {rate:.2f} clip/s | còn ~{eta / 60:.1f} phút", flush=True)
            t_log = time.time()
    if not feats:
        raise RuntimeError("không đọc được frame nào")
    out = torch.cat(feats).cpu().numpy().astype(np.float32, copy=False)
    if not np.isfinite(out).all():
        raise RuntimeError(f"feature có NaN/inf (tràn số với dtype={args.dtype}) -> thử --dtype bf16")
    return out


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    if device.type == "cuda":
        # fp32: tắt TF32 (mặc định bật cho conv trên GPU Ampere) để tính đúng fp32
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = True

    ffmpeg = find_ffmpeg()
    ids = shard(list_video_ids(args.video_dir, args.ids_from, args.video_ext), args.num_shards, args.shard_id)
    if args.limit:
        ids = ids[: args.limit]

    def out_path(vid: str) -> str:
        return os.path.join(args.output_dir, f"{vid}_one_peace_video_finetune.npy")

    todo = [v for v in ids if args.overwrite or not os.path.exists(out_path(v))]
    print(
        f"{len(ids)} video trong shard {args.shard_id}/{args.num_shards}, còn {len(todo)} video cần xử lý",
        flush=True,
    )
    if not todo:
        return

    t0 = time.time()
    model = load_k400_checkpoint(args.checkpoint, use_sdpa=not args.no_sdpa)
    model = model.to(device=device, dtype=dtype)
    print(
        f"Nạp model xong ({time.time() - t0:.0f}s), num_frames={model.num_frames}, dtype={dtype}", flush=True
    )
    if args.num_frames != model.num_frames:
        raise ValueError(f"checkpoint dùng {model.num_frames} frame / clip, không phải {args.num_frames}")
    compile_model = args.compile and device.type == "cuda"
    encode = build_encoder(model, device, dtype, args.batch_size, compile_model)

    fail_log = os.path.join(args.output_dir, f"failed_video_shard{args.shard_id}.txt")
    total_clips, t_all = 0, time.time()
    for i, vid in enumerate(todo):
        tag = f"[{i + 1}/{len(todo)}] {vid}"
        t_vid = time.time()
        try:
            feats = extract_one(os.path.join(args.video_dir, vid + args.video_ext), encode, args, ffmpeg, tag)
            save_npy_atomic(out_path(vid), feats)
        except Exception as e:
            print(f"[lỗi] {vid}: {e}", flush=True)
            with open(fail_log, "a", encoding="utf-8") as f:
                f.write(f"{vid}\t{e!r}\n")
            if isinstance(e, torch.cuda.OutOfMemoryError):
                traceback.print_exc()
                sys.exit(1)
            continue
        total_clips += len(feats)
        elapsed = time.time() - t_vid
        print(
            f"{tag}: XONG {feats.shape} trong {elapsed / 60:.1f} phút ({len(feats) / elapsed:.2f} clip/s) "
            f"| TB {total_clips / (time.time() - t_all):.2f} clip/s",
            flush=True,
        )


if __name__ == "__main__":
    main()
