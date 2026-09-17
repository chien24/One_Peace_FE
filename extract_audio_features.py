"""Trích xuất đặc trưng AUDIO cho UniAV bằng ONE-PEACE audio encoder.

Tái tạo cách tác giả UniAV làm:
  audio của video -> mono 16 kHz -> cửa sổ 1 s, bước `--stride_sec`
  -> mỗi cửa sổ xử lý như OnePeaceHubInterface.process_audio (layer_norm waveform)
  -> model.extract_audio_features (CLS -> audio_proj -> L2 normalize) -> 1536-d.
Kết quả: <output_dir>/<video_id>_one_peace_audio.npy, shape (T, 1536), float32.
(Feature audio DESED của tác giả cũng có chuẩn L2 = 1.)

Bước thời gian phải khớp visual: stride_sec = stride_frame / 16
  (0.25 s <-> stride 4 frame, 0.5 s <-> stride 8 frame).

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

import numpy as np
import torch
import torch.nn.functional as F

from video_io import find_ffmpeg, list_video_ids, load_audio, num_windows, probe_duration, save_npy_atomic, shard

SR = 16000


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--onepeace_repo", required=True, help="thư mục clone của https://github.com/OFA-Sys/ONE-PEACE")
    p.add_argument("--checkpoint", required=True, help="one-peace.pt hoặc one-peace-audio.pt")
    p.add_argument("--video_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--ids_from", default=None)
    p.add_argument("--video_ext", default=".mp4")
    p.add_argument("--window_sec", type=float, default=1.0)
    p.add_argument("--stride_sec", type=float, default=0.5)
    p.add_argument("--batch_size", type=int, default=64, help="số cửa sổ 1 s / lần forward")
    p.add_argument("--dtype", choices=["float32", "fp16", "bf16"], default="float32",
                   help="float32 = sát bản gốc nhất (mặc định)")
    p.add_argument("--resampler", choices=["librosa", "ffmpeg"], default="librosa",
                   help="librosa = giống librosa.load(sr=16000) trong process_audio gốc (mặc định)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--shard_id", type=int, default=0)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def load_onepeace_audio_model(repo: str, checkpoint: str, device: str, dtype: str):
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


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    window, hop = int(round(args.window_sec * SR)), int(round(args.stride_sec * SR))

    ffmpeg = find_ffmpeg()
    ids = shard(list_video_ids(args.video_dir, args.ids_from, args.video_ext), args.num_shards, args.shard_id)
    if args.limit:
        ids = ids[:args.limit]
    out_name = lambda vid: os.path.join(args.output_dir, f"{vid}_one_peace_audio.npy")
    todo = [v for v in ids if args.overwrite or not os.path.exists(out_name(v))]
    print(f"{len(ids)} video trong shard {args.shard_id}/{args.num_shards}, còn {len(todo)} video cần xử lý")
    if not todo:
        return

    t0 = time.time()
    hub = load_onepeace_audio_model(args.onepeace_repo, args.checkpoint, args.device, args.dtype)
    if args.dtype == "float32" and args.device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    T = hub._get_mask_indices_dims(window, hub.feature_encoder_spec)
    print(f"Nạp model xong ({time.time() - t0:.0f}s)")

    fail_log = os.path.join(args.output_dir, f"failed_audio_shard{args.shard_id}.txt")
    silent_log = os.path.join(args.output_dir, f"no_audio_track_shard{args.shard_id}.txt")
    for i, vid in enumerate(todo):
        path = os.path.join(args.video_dir, vid + args.video_ext)
        t_vid = time.time()
        try:
            wav = load_audio(path, SR, ffmpeg, args.resampler)
            if wav is None:
                # video không có âm thanh: dùng im lặng cùng độ dài để số bước khớp visual
                duration = probe_duration(path, ffmpeg) or 0.0
                wav = np.zeros(int(duration * SR), dtype=np.float32)
                with open(silent_log, "a", encoding="utf-8") as f:
                    f.write(vid + "\n")
            windows = make_windows(wav, window, hop)
            feats = []
            with torch.inference_mode():
                for b in range(0, len(windows), args.batch_size):
                    src = hub.cast_data_dtype(windows[b:b + args.batch_size].to(args.device))
                    masks = torch.zeros(src.size(0), T + 1, dtype=torch.bool, device=args.device)
                    feats.append(hub.extract_audio_features(src, masks).float().cpu().numpy())
            feats = np.concatenate(feats, axis=0).astype(np.float32)
            save_npy_atomic(out_name(vid), feats)
        except Exception as e:
            print(f"[lỗi] {vid}: {e}")
            with open(fail_log, "a", encoding="utf-8") as f:
                f.write(f"{vid}\t{repr(e)}\n")
            if isinstance(e, torch.cuda.OutOfMemoryError):
                sys.exit(1)
            continue
        print(f"[{i + 1}/{len(todo)}] {vid}: {feats.shape} trong {time.time() - t_vid:.1f}s")


if __name__ == "__main__":
    main()
