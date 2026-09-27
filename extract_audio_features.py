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

Chạy tiếp sau khi bị ngắt: video đã có .npy trong --output_dir, hoặc có tên trong done_audio*.txt
(script tự ghi vào --done_dir, mặc định = --output_dir) / các file, thư mục truyền qua --done_list,
sẽ được bỏ qua. Nhiều GPU (Kaggle 2xT4): mặc định dùng hết (--gpus auto), mỗi GPU một tiến trình
nhận một phần danh sách.

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
    append_line,
    auto_workers,
    find_ffmpeg,
    iter_parallel,
    list_video_ids,
    load_audio,
    load_audio_file,
    load_done_ids,
    num_windows,
    probe_duration,
    resolve_gpus,
    run_on_gpus,
    save_npy_atomic,
    shard,
)

SR = 16000
OUT_SUFFIX = "_one_peace_audio.npy"
DONE_PREFIX = "done_audio"


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
    p.add_argument(
        "--gpus",
        default="auto",
        help="auto = mọi GPU (Kaggle 2xT4 -> 2 tiến trình song song); '0' hoặc '0,1' để chọn; cpu = chạy CPU",
    )
    p.add_argument(
        "--decode_workers",
        type=int,
        default=0,
        help="số tiến trình đọc + resample audio cho mỗi GPU (0 = tự chọn theo số CPU)",
    )
    p.add_argument(
        "--done_dir",
        default=None,
        help="thư mục ghi/đọc file done_*.txt (mặc định = --output_dir). Colab: để trên Drive (file nhỏ) "
        "trong khi .npy ghi ở /content, phiên sau vẫn biết video nào đã xong",
    )
    p.add_argument(
        "--done_list",
        nargs="*",
        default=[],
        help="thêm nguồn 'đã xong' ngoài --output_dir: file .txt (mỗi dòng một id) hoặc thư mục .npy",
    )
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


def decode_audio(
    vid: str, args: argparse.Namespace
) -> Iterator[tuple[str, np.ndarray | None, bool, str | None]]:
    """(video_id, waveform 16 kHz, không có track audio?, lỗi nếu có) — chạy trong tiến trình giải mã."""
    src_dir, src_ext = (
        (args.audio_dir, args.audio_ext) if args.audio_dir else (args.video_dir, args.video_ext)
    )
    path = os.path.join(src_dir, args.file_prefix + vid + src_ext)
    try:
        ffmpeg = None if args.audio_dir else find_ffmpeg()
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
        yield vid, None, False, repr(e)


def run_gpu(todo: list[str], device_name: str, rank: int, args: argparse.Namespace, lock=None) -> None:
    """Xử lý ``todo`` trên một thiết bị. Chạy trong tiến trình riêng khi có nhiều GPU."""
    device = device_name
    if device.startswith("cuda"):
        torch.cuda.set_device(device)
    window, hop = int(round(args.window_sec * SR)), int(round(args.stride_sec * SR))
    prefix = f"[{device}] " if args.n_gpus > 1 else ""

    dtype = args.dtype
    if dtype == "auto":
        dtype = "fp16" if device.startswith("cuda") else "float32"
    t0 = time.time()
    if lock is not None:
        lock.acquire()  # nạp checkpoint lần lượt: mỗi bản chiếm ~6 GB RAM lúc nạp
    try:
        hub = load_onepeace_audio_model(args.onepeace_repo, args.checkpoint, device, dtype)
    finally:
        if lock is not None:
            lock.release()
    if dtype == "float32" and device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    T = hub._get_mask_indices_dims(window, hub.feature_encoder_spec)
    workers = args.decode_workers or auto_workers(args.n_gpus)
    print(
        f"{prefix}Nạp model xong ({time.time() - t0:.0f}s), dtype={dtype}, {len(todo)} video, "
        f"{workers} tiến trình giải mã",
        flush=True,
    )

    tag_id = f"shard{args.shard_id}" + (f"_gpu{rank}" if args.n_gpus > 1 else "")
    fail_log = os.path.join(args.output_dir, f"failed_audio_{tag_id}.txt")
    silent_log = os.path.join(args.output_dir, f"no_audio_track_{tag_id}.txt")
    done_file = os.path.join(args.done_dir, f"{DONE_PREFIX}_{tag_id}.txt")
    # giải mã + resample (librosa kaiser_best khá nặng CPU) ở nhiều tiến trình, mỗi tiến trình một video
    source = iter_parallel(functools.partial(decode_audio, args=args), todo, workers, max_queue=2 * workers)
    t_all, t_wait, n_win = time.time(), 0.0, 0
    try:
        for i in range(len(todo)):
            t = time.time()
            item = next(source, None)
            t_wait += time.time() - t
            if item is None:
                break
            if item[0] == "__error__":  # lỗi ngoài dự kiến trong tiến trình giải mã
                item = (item[1], None, False, item[2])
            vid, wav, silent, error = item
            t_vid = time.time()
            try:
                if error is not None:
                    raise RuntimeError(error)
                if silent:
                    append_line(silent_log, vid)
                windows = make_windows(wav, window, hop)
                chunks = []
                with torch.inference_mode():
                    for b in range(0, len(windows), args.batch_size):
                        src = hub.cast_data_dtype(windows[b : b + args.batch_size].to(device))
                        masks = torch.zeros(src.size(0), T + 1, dtype=torch.bool, device=device)
                        chunks.append(hub.extract_audio_features(src, masks).float())
                feats = torch.cat(chunks).cpu().numpy().astype(np.float32, copy=False)
                if not np.isfinite(feats).all():
                    raise RuntimeError(
                        f"feature có NaN/inf (tràn số với dtype={dtype}) -> thử --dtype float32"
                    )
                save_npy_atomic(os.path.join(args.output_dir, vid + OUT_SUFFIX), feats)
                append_line(done_file, vid)
            except Exception as e:
                print(f"{prefix}[lỗi] {vid}: {e}", flush=True)
                append_line(fail_log, f"{vid}\t{e!r}")
                if isinstance(e, torch.cuda.OutOfMemoryError):
                    sys.exit(1)
                continue
            n_win += len(feats)
            elapsed = time.time() - t_vid
            print(
                f"{prefix}[{i + 1}/{len(todo)}] {vid}: {feats.shape} trong {elapsed:.1f}s "
                f"({len(feats) / elapsed:.1f} cửa sổ/s)",
                flush=True,
            )
    finally:
        source.close()
    el = max(time.time() - t_all, 1e-9)
    print(
        f"{prefix}Hết: {n_win} cửa sổ trong {el / 60:.1f} phút ({n_win / el:.1f} cửa sổ/s, "
        f"chờ giải mã {100 * t_wait / el:.0f}% thời gian)",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    args.done_dir = args.done_dir or args.output_dir
    os.makedirs(args.done_dir, exist_ok=True)
    gpus = resolve_gpus(args.gpus)
    args.n_gpus = max(len(gpus), 1)

    src_dir, src_ext = (
        (args.audio_dir, args.audio_ext) if args.audio_dir else (args.video_dir, args.video_ext)
    )
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
    done = (
        set()
        if args.overwrite
        else load_done_ids(args.output_dir, OUT_SUFFIX, DONE_PREFIX, [args.done_dir, *args.done_list])
    )
    todo = [v for v in ids if v not in done]
    print(
        f"{len(ids)} video trong shard {args.shard_id}/{args.num_shards}, đã xong {len(ids) - len(todo)}, "
        f"còn {len(todo)} video cần xử lý | GPU: {gpus or 'không có (CPU)'}",
        flush=True,
    )
    if todo:
        run_on_gpus(run_gpu, todo, gpus, args)


if __name__ == "__main__":
    main()
