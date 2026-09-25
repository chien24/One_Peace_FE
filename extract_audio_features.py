"""Trích xuất đặc trưng AUDIO cho UniAV bằng ONE-PEACE audio encoder.

Tái tạo cách tác giả UniAV làm:
  audio của video -> mono 16 kHz -> cửa sổ 1 s, bước `--stride_sec`
  -> mỗi cửa sổ xử lý như OnePeaceHubInterface.process_audio (layer_norm waveform)
  -> model.extract_audio_features (CLS -> audio_proj -> L2 normalize) -> 1536-d.
Kết quả: <output_dir>/<video_id>_one_peace_audio.npy, shape (T, 1536), float32.
(Feature audio DESED của tác giả cũng có chuẩn L2 = 1.)

Bước thời gian phải khớp visual: stride_sec = stride_frame / 16
  (0.25 s <-> stride 4 frame, 0.5 s <-> stride 8 frame).

Nguồn audio (chọn một):
  --video_dir <thư mục mp4>  : (mặc định dùng) lấy track âm thanh trong mp4, giải mã + resample
                               trùng tuyệt đối với librosa.load(sr=16000) của code gốc.
  --audio_dir <thư mục wav>  : đọc <video_id>.wav bằng librosa.load(sr=16000). Chỉ sát bản gốc nếu wav
                               giữ sample rate gốc hoặc được resample bằng librosa
                               (không phải ffmpeg -ar 16000).

Cần repo ONE-PEACE + fairseq đi kèm (xem README.md), checkpoint one-peace.pt hoặc bản
đã tách bằng slim_audio_checkpoint.py.

  python extract_audio_features.py --onepeace_repo /content/ONE-PEACE \
      --checkpoint /content/one-peace-audio.pt --video_dir /content/YouCookII/videos \
      --output_dir /content/feats/youcookii --ids_from youcookii_train_preprocess.json --stride_sec 0.5
"""

import argparse
import functools
import math
import os
import sys
import time
from collections.abc import Iterator
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from video_io import (
    find_ffmpeg,
    list_video_ids,
    load_audio,
    load_audio_file,
    num_windows,
    prefetch,
    probe_duration,
    save_npy_atomic,
    shard,
)

SR = 16000


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--onepeace_repo", required=True, help="thư mục clone của https://github.com/OFA-Sys/ONE-PEACE"
    )
    p.add_argument("--checkpoint", required=True, help="one-peace.pt hoặc one-peace-audio.pt")
    p.add_argument("--video_dir", default=None, help="thư mục mp4 (khi audio nằm trong video)")
    p.add_argument("--audio_dir", default=None, help="thư mục <video_id>.wav (khi đã tách audio riêng)")
    p.add_argument("--audio_ext", default=".wav")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--ids_from", default=None)
    p.add_argument("--video_ext", default=".mp4")
    p.add_argument("--window_sec", type=float, default=1.0)
    p.add_argument("--stride_sec", type=float, default=0.5)
    p.add_argument("--batch_size", type=int, default=64, help="số cửa sổ 1 s / lần forward")
    p.add_argument(
        "--dtype",
        choices=["auto", "float32", "fp16", "bf16"],
        default="auto",
        help="auto = fp16 trên GPU (nhanh ~2.6 lần, cos 0.999996 so với float32), float32 trên CPU",
    )
    p.add_argument(
        "--res_type",
        choices=["kaiser_best", "kaiser_fast", "soxr_hq", "soxr_vhq"],
        default="kaiser_best",
        help="bộ lọc resample của librosa. kaiser_best (mặc định của librosa < 0.10) khớp nhất với "
        "feature DESED của tác giả UniAV: cos 0.93 so với 0.84 của soxr_hq (README mục 9); cần gói resampy",
    )
    p.add_argument(
        "--file_prefix",
        default="",
        help="tiền tố tên file mà id trong annotation không có, ví dụ 'Y' cho tập validation của DESED "
        "(id <id> ứng với file Y<id>.wav)",
    )
    p.add_argument(
        "--resampler",
        choices=["librosa", "ffmpeg"],
        default="librosa",
        help="librosa = giống librosa.load(sr=16000) trong process_audio gốc (mặc định)",
    )
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--shard_id", type=int, default=0)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    if bool(args.video_dir) == bool(args.audio_dir):
        p.error("chọn đúng một nguồn audio: --video_dir hoặc --audio_dir")
    return args


def load_onepeace_audio_model(repo: str, checkpoint: str, device: str, dtype: str) -> Any:
    """Giống one_peace.models.from_pretrained(...), nhưng chỉ dựng nhánh audio (head_type='audio')
    và trỏ bpe_dir về repo để chạy được từ thư mục bất kỳ."""
    repo = os.path.abspath(repo)
    sys.path.insert(0, repo)
    # ONE-PEACE/fairseq/ phải đứng trước repo, nếu không `fairseq` bị hiểu là namespace package rỗng
    sys.path.insert(0, os.path.join(repo, "fairseq"))

    # checkpoint fairseq chứa cfg/argparse -> torch >= 2.6 cần weights_only=False
    torch.load = functools.partial(torch.load, weights_only=False)

    from fairseq import checkpoint_utils, tasks
    from one_peace.models.one_peace.hub_interface import OnePeaceHubInterface

    # cfg trong one-peace.pt: model._name = 'one_piece_pretrain' (gõ sai, không đăng ký),
    # head_type = 'val', bpe_dir tương đối -> override như from_pretrained gốc
    overrides = {
        "model": {"_name": "one_peace_retrieval"},
        "task": {"head_type": "audio", "bpe_dir": os.path.join(repo, "one_peace", "utils", "BPE")},
    }
    state = checkpoint_utils.load_checkpoint_to_cpu(checkpoint, arg_overrides=overrides)
    cfg = state["cfg"]
    task = tasks.setup_task(cfg.task)

    # Khác load_model_ensemble_and_task: dựng model trên 'meta' rồi gán thẳng tensor của checkpoint,
    # tránh cấp phát thêm ~6 GB tham số fp32 ngẫu nhiên (hết RAM trên máy 16 GB / Colab thường).
    # TransformerEncoder gọi torch.linspace(...).item() cho drop-path -> để hàm đó chạy trên CPU
    linspace = torch.linspace
    torch.linspace = lambda *a, **kw: linspace(*a, **{**kw, "device": "cpu"})
    try:
        with torch.device("meta"):
            # from_checkpoint=True: bỏ các khoá cfg chỉ có ở bản pretrain (reset_logit_scale, stage2_pretrain)
            model = task.build_model(cfg.model, from_checkpoint=True)
    finally:
        torch.linspace = linspace
    state_dict = state.pop("model")
    model.remove_pretraining_modules(state_dict)
    if "logit_scale" not in state_dict:
        # one-peace.pt không lưu logit_scale (cfg reset_logit_scale=True); bản gốc cũng khởi tạo lại.
        # Tham số này chỉ dùng cho độ tương đồng audio-text, không ảnh hưởng feature.
        state_dict["logit_scale"] = torch.tensor(math.log(1 / 0.07))
    missing, unexpected = torch.nn.Module.load_state_dict(model, state_dict, strict=False, assign=True)
    if missing or unexpected:
        raise RuntimeError(f"checkpoint không khớp model: thiếu {missing[:10]}, thừa {unexpected[:10]}")
    del state, state_dict
    return OnePeaceHubInterface(cfg, task, model, device=device, dtype=dtype)


def make_windows(wav: np.ndarray, window: int, hop: int) -> torch.Tensor:
    """(N, window) — mỗi cửa sổ đã layer_norm như process_audio của ONE-PEACE."""
    wav = torch.from_numpy(wav)
    if wav.numel() < window:  # process_audio: audio < 1 s thì lặp lại cho đủ
        wav = wav.repeat(math.ceil(window / max(wav.numel(), 1)))[:window]
    n = num_windows(wav.numel(), window, hop)
    chunks = wav.unfold(0, window, hop)[:n]  # N x window
    return F.layer_norm(chunks, (window,))


def iter_audio(
    todo: list[str], src_dir: str, src_ext: str, args: argparse.Namespace, ffmpeg: str | None
) -> Iterator[tuple[str, np.ndarray | None, bool, Exception | None]]:
    """(video_id, waveform 16 kHz, không có track audio?, lỗi nếu có) — chạy ở thread nền."""
    for vid in todo:
        path = os.path.join(src_dir, args.file_prefix + vid + src_ext)
        try:
            wav = (
                load_audio_file(path, SR, args.res_type)
                if args.audio_dir
                else load_audio(path, SR, ffmpeg, args.resampler, args.res_type)
            )
            silent = wav is None
            if silent:
                # video không có âm thanh: dùng im lặng cùng độ dài để số bước khớp visual
                duration = probe_duration(path, ffmpeg or find_ffmpeg()) or 0.0
                wav = np.zeros(int(duration * SR), dtype=np.float32)
            yield vid, wav, silent, None
        except Exception as e:  # lỗi của một video không làm dừng cả shard
            yield vid, None, False, e


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    window, hop = int(round(args.window_sec * SR)), int(round(args.stride_sec * SR))

    src_dir, src_ext = (
        (args.audio_dir, args.audio_ext) if args.audio_dir else (args.video_dir, args.video_ext)
    )
    ffmpeg = None if args.audio_dir else find_ffmpeg()
    all_ids = list_video_ids(src_dir, args.ids_from, src_ext, file_prefix=args.file_prefix)
    if args.ids_from:
        wanted = list_video_ids(src_dir, args.ids_from, src_ext, must_exist=False)
        missing = sorted(set(wanted) - set(all_ids))
        if missing:
            print(
                f"[warn] {len(missing)} video trong {args.ids_from} không có file "
                f"{args.file_prefix}<id>{src_ext} trong {src_dir}, "
                f"ví dụ: {missing[:5]}"
            )
    ids = shard(all_ids, args.num_shards, args.shard_id)
    if args.limit:
        ids = ids[: args.limit]
    out_name = lambda vid: os.path.join(args.output_dir, f"{vid}_one_peace_audio.npy")
    todo = [v for v in ids if args.overwrite or not os.path.exists(out_name(v))]
    print(f"{len(ids)} video trong shard {args.shard_id}/{args.num_shards}, còn {len(todo)} video cần xử lý")
    if not todo:
        return

    dtype = args.dtype
    if dtype == "auto":
        dtype = "fp16" if args.device.startswith("cuda") else "float32"
    t0 = time.time()
    hub = load_onepeace_audio_model(args.onepeace_repo, args.checkpoint, args.device, dtype)
    if dtype == "float32" and args.device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    T = hub._get_mask_indices_dims(window, hub.feature_encoder_spec)
    print(f"Nạp model xong ({time.time() - t0:.0f}s), dtype={dtype}", flush=True)

    fail_log = os.path.join(args.output_dir, f"failed_audio_shard{args.shard_id}.txt")
    silent_log = os.path.join(args.output_dir, f"no_audio_track_shard{args.shard_id}.txt")
    # đọc audio của video kế tiếp ở thread nền, GPU không phải chờ giải mã
    source = prefetch(iter_audio(todo, src_dir, src_ext, args, ffmpeg), max_items=1)
    for i, (vid, wav, silent, error) in enumerate(source):
        t_vid = time.time()
        try:
            if error is not None:
                raise error
            if silent:
                with open(silent_log, "a", encoding="utf-8") as f:
                    f.write(vid + "\n")
            windows = make_windows(wav, window, hop)
            chunks = []
            with torch.inference_mode():
                for b in range(0, len(windows), args.batch_size):
                    src = hub.cast_data_dtype(windows[b : b + args.batch_size].to(args.device))
                    masks = torch.zeros(src.size(0), T + 1, dtype=torch.bool, device=args.device)
                    chunks.append(hub.extract_audio_features(src, masks).float())
            feats = torch.cat(chunks).cpu().numpy().astype(np.float32, copy=False)
            if not np.isfinite(feats).all():
                raise RuntimeError(f"feature có NaN/inf (tràn số với dtype={dtype}) -> thử --dtype float32")
            save_npy_atomic(out_name(vid), feats)
        except Exception as e:
            print(f"[lỗi] {vid}: {e}", flush=True)
            with open(fail_log, "a", encoding="utf-8") as f:
                f.write(f"{vid}\t{e!r}\n")
            if isinstance(e, torch.cuda.OutOfMemoryError):
                sys.exit(1)
            continue
        elapsed = time.time() - t_vid
        print(
            f"[{i + 1}/{len(todo)}] {vid}: {feats.shape} trong {elapsed:.1f}s "
            f"({len(feats) / elapsed:.1f} cửa sổ/s)",
            flush=True,
        )


if __name__ == "__main__":
    main()
