"""Trích xuất đặc trưng VISUAL cho UniAV bằng ONE-PEACE video encoder (fine-tune K400).

Tái tạo cách tác giả UniAV làm (README + mục Implementation Details của bài báo):
  video -> 16 fps -> resize cạnh ngắn 256 -> center crop 256x256
        -> cửa sổ 16 frame, bước `--stride` frame
        -> ONE-PEACE video backbone -> trung bình CLS token 16 frame -> 1536-d.
Kết quả: <output_dir>/<video_id>_one_peace_video_finetune.npy, shape (T, 1536).

  stride 4 (0.25 s): DESED, UnAV-100      stride 8 (0.5 s): ActivityNet-1.3

Ví dụ (Colab):
  python extract_video_features.py --checkpoint onepeace_video_k400.pth \
      --video_dir /content/YouCookII/videos --output_dir /content/feats/youcookii \
      --ids_from youcookii_train_preprocess.json --stride 8 --batch_size 4
"""

import argparse
import os
import sys
import time
import traceback

import numpy as np
import torch

from onepeace_video_backbone import IMG_MEAN, IMG_STD, load_k400_checkpoint
from video_io import (find_ffmpeg, iter_clips, iter_video_frames, list_video_ids, prefetch,
                      save_npy_atomic, shard)


def parse_args():
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
    p.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="fp32",
                   help="fp32 = sát bản gốc nhất (mặc định); fp16/bf16 nhanh hơn ~2-3 lần, sai khác nhỏ")
    p.add_argument("--resize_backend", choices=["cv2", "ffmpeg"], default="cv2",
                   help="cv2 = resize/crop y hệt mmaction2 của ONE-PEACE (mặc định); ffmpeg = nhanh hơn")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--no_sdpa", action="store_true", help="dùng attention bmm gốc thay vì SDPA")
    p.add_argument("--num_shards", type=int, default=1, help="chia danh sách video cho nhiều phiên chạy")
    p.add_argument("--shard_id", type=int, default=0)
    p.add_argument("--limit", type=int, default=None, help="chỉ xử lý N video đầu (để thử)")
    p.add_argument("--max_clips", type=int, default=None, help="chỉ lấy N clip đầu mỗi video (để thử)")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    if device.type == "cpu" and dtype != torch.float32:
        print("[warn] CPU: chuyển sang fp32")
        dtype = torch.float32
    if dtype == torch.float32 and device.type == "cuda":
        # tắt TF32 (mặc định bật cho conv trên GPU Ampere) để tính đúng fp32
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    ffmpeg = find_ffmpeg()
    ids = shard(list_video_ids(args.video_dir, args.ids_from, args.video_ext), args.num_shards, args.shard_id)
    if args.limit:
        ids = ids[:args.limit]
    out_name = lambda vid: os.path.join(args.output_dir, f"{vid}_one_peace_video_finetune.npy")
    todo = [v for v in ids if args.overwrite or not os.path.exists(out_name(v))]
    print(f"{len(ids)} video trong shard {args.shard_id}/{args.num_shards}, còn {len(todo)} video cần xử lý")
    if not todo:
        return

    t0 = time.time()
    model = load_k400_checkpoint(args.checkpoint, use_sdpa=not args.no_sdpa)
    model = model.to(device=device, dtype=dtype)
    print(f"Nạp model xong ({time.time() - t0:.0f}s), num_frames={model.num_frames}, dtype={dtype}")
    assert args.num_frames == model.num_frames, "checkpoint K400 dùng đúng 16 frame / clip"

    mean = torch.tensor(IMG_MEAN, device=device, dtype=torch.float32).view(1, 3, 1, 1, 1)
    std = torch.tensor(IMG_STD, device=device, dtype=torch.float32).view(1, 3, 1, 1, 1)

    @torch.inference_mode()
    def encode(clips):
        x = torch.from_numpy(np.stack(clips)).to(device, non_blocking=True)  # B T H W 3 uint8
        x = x.permute(0, 4, 1, 2, 3).float()                                  # B 3 T H W
        x = ((x - mean) / std).to(dtype)
        return model.extract_clip_features(x).float().cpu().numpy()

    fail_log = os.path.join(args.output_dir, f"failed_video_shard{args.shard_id}.txt")
    total_clips, t_start = 0, time.time()
    for i, vid in enumerate(todo):
        path = os.path.join(args.video_dir, vid + args.video_ext)
        t_vid = time.time()
        try:
            clips_iter = iter_clips(iter_video_frames(path, args.fps, args.size, ffmpeg, args.resize_backend),
                                    args.num_frames, args.stride)
            feats, batch = [], []
            for j, clip in enumerate(prefetch(clips_iter, max_items=args.batch_size * 3)):
                if args.max_clips is not None and j >= args.max_clips:
                    break
                batch.append(clip)
                if len(batch) == args.batch_size:
                    feats.append(encode(batch))
                    batch = []
            if batch:
                feats.append(encode(batch))
            if not feats:
                raise RuntimeError("không đọc được frame nào")
            feats = np.concatenate(feats, axis=0).astype(np.float32)
            save_npy_atomic(out_name(vid), feats)
        except Exception as e:
            print(f"[lỗi] {vid}: {e}")
            with open(fail_log, "a", encoding="utf-8") as f:
                f.write(f"{vid}\t{repr(e)}\n")
            if isinstance(e, torch.cuda.OutOfMemoryError):
                traceback.print_exc()
                sys.exit(1)
            continue
        total_clips += len(feats)
        speed = total_clips / (time.time() - t_start)
        print(f"[{i + 1}/{len(todo)}] {vid}: {feats.shape} trong {time.time() - t_vid:.1f}s "
              f"| TB {speed:.2f} clip/s")


if __name__ == "__main__":
    main()
