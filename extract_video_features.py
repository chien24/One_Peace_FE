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

Chạy tiếp sau khi bị ngắt: video đã có .npy trong --output_dir, hoặc có tên trong done_video*.txt
(script tự ghi vào --done_dir, mặc định = --output_dir) / các file, thư mục truyền qua --done_list,
sẽ được bỏ qua. Nhiều GPU (Kaggle 2xT4): mặc định dùng hết (--gpus auto), mỗi GPU một tiến trình
nhận một phần danh sách.
"""

from __future__ import annotations

import argparse
import functools
import os
import queue
import sys
import threading
import time
import traceback
from collections.abc import Callable, Iterator

import numpy as np
import torch

from onepeace_video_backbone import IMG_MEAN, IMG_STD, OnePeaceViT, load_k400_checkpoint
from video_io import (
    append_line,
    auto_workers,
    find_ffmpeg,
    iter_clip_chunks,
    iter_parallel,
    iter_video_frames,
    list_video_ids,
    load_done_ids,
    num_windows,
    probe,
    resolve_gpus,
    run_on_gpus,
    save_npy_atomic,
    shard,
)

OUT_SUFFIX = "_one_peace_video_finetune.npy"
DONE_PREFIX = "done_video"

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
    p.add_argument(
        "--gpus",
        default="auto",
        help="auto = mọi GPU (Kaggle 2xT4 -> 2 tiến trình song song); '0' hoặc '0,1' để chọn; cpu = chạy CPU",
    )
    p.add_argument(
        "--decode_workers",
        type=int,
        default=0,
        help="số tiến trình giải mã video cho mỗi GPU, mỗi tiến trình một video (0 = tự chọn theo số CPU)",
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
    p.add_argument("--no_sdpa", action="store_true", help="dùng attention matmul gốc thay vì SDPA")
    p.add_argument(
        "--compile",
        action="store_true",
        help="(GPU) torch.compile: nhanh hơn ~7%% trên RTX 3060, nhiều hơn trên A100; mất ~1 phút "
        "biên dịch lúc đầu. Linux/Colab dùng được ngay; Windows cần `pip install triton-windows`",
    )
    p.add_argument("--num_shards", type=int, default=1, help="chia danh sách video cho nhiều phiên chạy")
    p.add_argument("--shard_id", type=int, default=0)
    p.add_argument(
        "--limit", type=int, default=None, help="lần chạy này chỉ xử lý N video chưa xong rồi dừng"
    )
    p.add_argument("--max_clips", type=int, default=None, help="chỉ lấy N clip đầu mỗi video (để thử)")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--log_every", type=float, default=60, help="in tiến độ mỗi N giây")
    return p.parse_args()


def resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    if device.type == "cpu":
        if name not in ("auto", "fp32"):
            print("[warn] CPU: chuyển sang fp32")
        return torch.float32
    return torch.float16 if name == "auto" else DTYPES[name]


def build_encoder(
    model: OnePeaceViT,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int,
    num_frames: int,
    stride: int,
    compile_model: bool,
) -> Callable[[np.ndarray, int], torch.Tensor]:
    """Hàm (khối frame (L, H, W, 3) uint8, số clip m) -> (m, 1536) float32 trên ``device``.

    Khối frame từ ``iter_clip_chunks``; clip được dựng lại trên GPU bằng ``unfold`` (không sao chép
    trên CPU). Kết quả chưa đồng bộ về CPU, GPU chạy nối tiếp trong khi CPU chuẩn bị khối sau.
    """
    mean = torch.tensor(IMG_MEAN, device=device).view(1, 3, 1, 1, 1)
    std = torch.tensor(IMG_STD, device=device).view(1, 3, 1, 1, 1)
    pin = device.type == "cuda"
    forward = model.extract_clip_features
    if compile_model:
        forward = torch.compile(forward, dynamic=False)

    @torch.inference_mode()
    def encode(chunk: np.ndarray, m: int) -> torch.Tensor:
        x = torch.from_numpy(chunk)
        if pin:
            x = x.pin_memory()
        x = x.to(device, non_blocking=True)
        clips = x.unfold(0, num_frames, stride)[:m].permute(0, 3, 4, 1, 2)  # m 3 T H W
        if compile_model and m < batch_size:
            # batch cuối của video: lặp clip cuối cho đủ batch, tránh biên dịch lại cho shape mới
            clips = torch.cat([clips, clips[-1:].expand(batch_size - m, *clips.shape[1:])])
        x = ((clips.float() - mean) / std).to(dtype)
        return forward(x)[:m].float()

    return encode


def decode_video(vid: str, args: argparse.Namespace) -> Iterator[tuple]:
    """Chạy trong tiến trình giải mã: ('start', vid, thời lượng, số clip dự kiến), rồi các
    ('chunk', vid, khối frame, số clip), cuối cùng ('end', vid, số clip, lỗi hoặc None)."""
    n = 0
    try:
        ffmpeg = find_ffmpeg()
        path = os.path.join(args.video_dir, vid + args.video_ext)
        info = probe(path, ffmpeg)
        duration = info["duration"] or 0.0
        n_expected = num_windows(int(duration * args.fps), args.num_frames, args.stride)
        if args.max_clips is not None:
            n_expected = min(n_expected, args.max_clips)
        yield ("start", vid, duration, n_expected)
        frames = iter_video_frames(path, args.fps, args.size, ffmpeg, args.resize_backend, info=info)
        for chunk, m in iter_clip_chunks(
            frames, args.num_frames, args.stride, args.batch_size, args.max_clips
        ):
            yield ("chunk", vid, chunk, m)
            n += m
        yield ("end", vid, n, None if n else "không đọc được frame nào")
    except Exception as e:
        yield ("end", vid, n, repr(e))


class Saver:
    """Thread nền: chép feature về CPU, kiểm tra, ghi .npy rồi ghi id vào done list.

    Ghi lên Drive có thể mất cả giây mỗi file; làm ở đây để GPU không phải chờ.
    """

    def __init__(self, device: torch.device, out_path: Callable[[str], str], done_file: str, fail_log: str):
        self.q: queue.Queue = queue.Queue(maxsize=8)
        self.device, self.out_path, self.done_file, self.fail_log = device, out_path, done_file, fail_log
        self.error: BaseException | None = None
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
        while (item := self.q.get()) is not None:
            vid, feats, tag, msg = item
            try:
                out = torch.cat(feats).cpu().numpy().astype(np.float32, copy=False)
                if not np.isfinite(out).all():
                    raise RuntimeError("feature có NaN/inf (tràn số fp16) -> thử --dtype bf16")
                save_npy_atomic(self.out_path(vid), out)
                append_line(self.done_file, vid)
                print(f"{tag}: XONG {out.shape}{msg}", flush=True)
            except Exception as e:
                print(f"[lỗi] {vid}: {e}", flush=True)
                append_line(self.fail_log, f"{vid}\t{e!r}")
            except BaseException as e:  # noqa: BLE001 - báo cho thread chính
                self.error = e
                return

    def put(self, *item) -> None:
        if self.error:
            raise self.error
        self.q.put(item)

    def close(self) -> None:
        self.q.put(None)
        self.thread.join()
        if self.error:
            raise self.error


def run_gpu(todo: list[str], device_name: str, rank: int, args: argparse.Namespace, lock=None) -> None:
    """Xử lý ``todo`` trên một thiết bị. Chạy trong tiến trình riêng khi có nhiều GPU."""
    device = torch.device(device_name)
    dtype = resolve_dtype(args.dtype, device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        # fp32: tắt TF32 (mặc định bật cho conv trên GPU Ampere) để tính đúng fp32
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = True
    prefix = f"[{device_name}] " if args.n_gpus > 1 else ""

    t0 = time.time()
    if lock is not None:
        lock.acquire()
    try:
        model = load_k400_checkpoint(args.checkpoint, use_sdpa=not args.no_sdpa)
        model = model.to(device=device, dtype=dtype)
    finally:
        if lock is not None:
            lock.release()
    if args.num_frames != model.num_frames:
        raise ValueError(f"checkpoint dùng {model.num_frames} frame / clip, không phải {args.num_frames}")
    compile_model = args.compile and device.type == "cuda"
    encode = build_encoder(model, device, dtype, args.batch_size, args.num_frames, args.stride, compile_model)
    workers = args.decode_workers or auto_workers(args.n_gpus)
    print(
        f"{prefix}Nạp model xong ({time.time() - t0:.0f}s), num_frames={model.num_frames}, dtype={dtype}, "
        f"{len(todo)} video, {workers} tiến trình giải mã",
        flush=True,
    )

    tag_id = f"shard{args.shard_id}" + (f"_gpu{rank}" if args.n_gpus > 1 else "")
    saver = Saver(
        device,
        lambda vid: os.path.join(args.output_dir, vid + OUT_SUFFIX),
        os.path.join(args.done_dir, f"{DONE_PREFIX}_{tag_id}.txt"),
        os.path.join(args.output_dir, f"failed_video_{tag_id}.txt"),
    )
    state: dict[str, dict] = {}  # video đang dở: feature trên GPU, thời điểm bắt đầu, ...
    n_done = total_clips = 0
    t_all = t_log = time.time()
    t_wait = 0.0  # thời gian vòng lặp GPU phải đứng chờ khâu giải mã
    source = iter_parallel(
        functools.partial(decode_video, args=args), todo, workers, max_queue=2 * workers + 2
    )
    try:
        while True:
            t = time.time()
            msg = next(source, None)
            t_wait += time.time() - t
            if msg is None:
                break
            kind, vid = msg[0], msg[1]
            if kind == "start":
                state[vid] = {"feats": [], "t0": time.time(), "n_expected": msg[3]}
                n_started = len(state) + n_done
                print(f"{prefix}[{n_started}/{len(todo)}] {vid}: {msg[2]:.0f}s ~ {msg[3]} clip", flush=True)
            elif kind == "chunk":
                st = state[vid]
                if st.get("error"):
                    continue
                try:
                    st["feats"].append(encode(msg[2], msg[3]))  # giữ trên GPU, không đồng bộ
                except torch.cuda.OutOfMemoryError:
                    raise
                except Exception as e:  # lỗi của một video không làm dừng cả shard
                    st["error"] = repr(e)
                    continue
                total_clips += msg[3]
            elif kind in ("end", "__error__"):
                st = state.pop(vid, None)
                n_done += 1
                error = (msg[3] or (st or {}).get("error")) if kind == "end" else msg[2]
                tag = f"{prefix}[{n_done}/{len(todo)}] {vid}"
                if error or st is None:
                    print(f"[lỗi] {vid}: {error}", flush=True)
                    append_line(saver.fail_log, f"{vid}\t{error}")
                    continue
                elapsed = time.time() - st["t0"]
                saver.put(vid, st["feats"], tag, f" trong {elapsed / 60:.1f} phút")
            if time.time() - t_log > args.log_every:
                el = time.time() - t_all
                print(
                    f"{prefix}    {n_done}/{len(todo)} video | TB {total_clips / el:.2f} clip/s "
                    f"| chờ giải mã {100 * t_wait / el:.0f}% thời gian",
                    flush=True,
                )
                t_log = time.time()
    except torch.cuda.OutOfMemoryError:
        traceback.print_exc()
        sys.exit(1)
    finally:
        source.close()
        saver.close()
    el = time.time() - t_all
    print(
        f"{prefix}Hết: {n_done} video, {total_clips} clip trong {el / 60:.1f} phút "
        f"({total_clips / max(el, 1e-9):.2f} clip/s, chờ giải mã {100 * t_wait / max(el, 1e-9):.0f}%)",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    args.done_dir = args.done_dir or args.output_dir
    os.makedirs(args.done_dir, exist_ok=True)
    gpus = resolve_gpus(args.gpus)
    args.n_gpus = max(len(gpus), 1)

    ids = shard(list_video_ids(args.video_dir, args.ids_from, args.video_ext), args.num_shards, args.shard_id)
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
    if args.limit and len(todo) > args.limit:
        todo = todo[: args.limit]
        print(f"--limit {args.limit}: lần này chỉ xử lý {len(todo)} video", flush=True)
    if todo:
        run_on_gpus(run_gpu, todo, gpus, args)


if __name__ == "__main__":
    main()
