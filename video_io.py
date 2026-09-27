"""Đọc video/audio bằng ffmpeg (stream, không bung toàn bộ video vào RAM) và các tiện ích chung."""

from __future__ import annotations

import json
import os
import queue
import re
import shutil
import subprocess
import tempfile
import threading
from collections.abc import Iterable, Iterator

import numpy as np

# --------------------------------------------------------------------------- ffmpeg


def find_ffmpeg() -> str:
    """Đường dẫn ffmpeg: ưu tiên bản trong PATH, sau đó tới bản của ``imageio-ffmpeg``."""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
    except ImportError:
        raise RuntimeError(
            "Không tìm thấy ffmpeg. Cài bằng `apt install ffmpeg` hoặc `pip install imageio-ffmpeg`."
        ) from None
    return imageio_ffmpeg.get_ffmpeg_exe()


def probe(path: str, ffmpeg: str | None = None) -> dict:
    """Thông tin cơ bản từ ``ffmpeg -i`` (không cần ffprobe): duration, width, height, sample_rate."""
    ffmpeg = ffmpeg or find_ffmpeg()
    proc = subprocess.run(
        [ffmpeg, "-hide_banner", "-noautorotate", "-i", path],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="ignore",
    )
    info = {"duration": None, "width": None, "height": None, "sample_rate": None}
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", proc.stderr)
    if m:
        h, mi, s = m.groups()
        info["duration"] = int(h) * 3600 + int(mi) * 60 + float(s)
    m = re.search(r"Stream #.*?Video:.*?\s(\d{2,5})x(\d{2,5})[\s,\[]", proc.stderr)
    if m:
        info["width"], info["height"] = int(m.group(1)), int(m.group(2))
    m = re.search(r"Stream #.*?Audio:.*?(\d+) Hz", proc.stderr)
    if m:
        info["sample_rate"] = int(m.group(1))
    return info


def probe_duration(path: str, ffmpeg: str | None = None) -> float | None:
    return probe(path, ffmpeg)["duration"]


def _run_ffmpeg_stream(cmd: list[str], chunk_bytes: int) -> Iterator[bytes]:
    """Chạy ffmpeg, trả về từng khối ``chunk_bytes`` byte từ stdout. Lỗi ffmpeg -> RuntimeError."""
    with tempfile.TemporaryFile() as err:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=err, bufsize=chunk_bytes * 4)
        try:
            while True:
                buf = proc.stdout.read(chunk_bytes)
                if len(buf) < chunk_bytes:
                    break
                yield buf
        finally:
            proc.stdout.close()
            ret = proc.wait()
        if ret != 0:
            err.seek(0)
            raise RuntimeError(f"ffmpeg lỗi ({ret}): {err.read().decode('utf-8', 'ignore')[-800:]}")


# --------------------------------------------------------------------------- video


def mmaction_resize_center_crop(img: np.ndarray, size: int = 256) -> np.ndarray:
    """Đúng pipeline test của ONE-PEACE K400 (mmaction2).

    ``Resize(scale=(-1, size))`` = ``mmcv.rescale_size`` + ``mmcv.imresize`` (cv2 INTER_LINEAR),
    rồi ``CenterCrop(size)``.
    """
    import cv2

    h, w = img.shape[:2]
    scale = size / min(h, w)  # rescale_size((w, h), (inf, size))
    new_w, new_h = int(w * scale + 0.5), int(h * scale + 0.5)
    if (new_w, new_h) != (w, h):
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    left, top = (new_w - size) // 2, (new_h - size) // 2  # CenterCrop
    return np.ascontiguousarray(img[top : top + size, left : left + size])


def iter_video_frames(
    path: str,
    fps: int = 16,
    size: int = 256,
    ffmpeg: str | None = None,
    resize_backend: str = "cv2",
    info: dict | None = None,
) -> Iterator[np.ndarray]:
    """Frame RGB uint8 (size, size, 3): lấy mẫu lại ``fps`` -> resize cạnh ngắn = size -> center crop.

    resize_backend='cv2' (mặc định, sát bản gốc nhất): ffmpeg chỉ giải mã + đổi fps ở độ phân giải gốc,
        resize/crop làm bằng cv2 y như mmaction2 (pipeline ONE-PEACE fine-tune K400).
    resize_backend='ffmpeg': resize/crop luôn trong ffmpeg (nhanh hơn, nội suy hơi khác).
    ``info``: kết quả ``probe(path)`` nếu đã có (tránh gọi ffmpeg thêm một lần).
    Không tự xoay video theo metadata (-noautorotate), giống decord mà mmaction2 dùng.
    """
    ffmpeg = ffmpeg or find_ffmpeg()
    base = [
        ffmpeg,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-noautorotate",
        "-i",
        path,
        "-an",
        "-sn",
    ]
    if resize_backend == "ffmpeg":
        vf = (
            f"fps={fps},"
            f"scale='if(lte(iw,ih),{size},-2)':'if(lte(iw,ih),-2,{size})':flags=bilinear,"
            f"crop={size}:{size}"
        )
        cmd = base + ["-vf", vf, "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1"]
        for buf in _run_ffmpeg_stream(cmd, size * size * 3):
            yield np.frombuffer(buf, dtype=np.uint8).reshape(size, size, 3)
        return

    info = info or probe(path, ffmpeg)
    w, h = info["width"], info["height"]
    if not w or not h:
        raise RuntimeError(f"không đọc được kích thước video: {path}")
    cmd = base + ["-vf", f"fps={fps}", "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1"]
    for buf in _run_ffmpeg_stream(cmd, w * h * 3):
        frame = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3)
        yield mmaction_resize_center_crop(frame, size)


def iter_clips(frames: Iterable[np.ndarray], num_frames: int = 16, stride: int = 4) -> Iterator[np.ndarray]:
    """Cửa sổ trượt ``num_frames`` frame, bước ``stride`` -> mảng (num_frames, H, W, 3).

    Số clip = floor((N - num_frames) / stride) + 1, clip i bắt đầu ở frame i * stride.
    Video ngắn hơn num_frames: lặp lại frame cuối để đủ 1 clip.
    """
    buf: list[np.ndarray] = []
    next_start = 0  # chỉ số frame bắt đầu của clip kế tiếp
    buf_start = 0  # chỉ số frame của buf[0]
    emitted = False
    for idx, frame in enumerate(frames):
        buf.append(frame)
        if idx == next_start + num_frames - 1:
            offset = next_start - buf_start
            yield np.stack(buf[offset : offset + num_frames])
            emitted = True
            next_start += stride
            removed = min(next_start - buf_start, len(buf))
            del buf[:removed]
            buf_start += removed
    if not emitted and buf:
        clip = buf + [buf[-1]] * (num_frames - len(buf))
        yield np.stack(clip[:num_frames])


def iter_clip_chunks(
    frames: Iterable[np.ndarray], num_frames: int, stride: int, clips_per_chunk: int, limit: int | None = None
) -> Iterator[tuple[np.ndarray, int]]:
    """Giống ``iter_batches(iter_clips(...))`` nhưng không nhân bản frame: trả về (khối frame, số clip m).

    Khối gồm (m - 1) * stride + num_frames frame liên tiếp; clip thứ k của khối là
    ``khối[k * stride : k * stride + num_frames]`` (dựng lại trên GPU bằng ``unfold``). Với stride 8,
    lượng dữ liệu phải chuyển chỉ còn khoảng một nửa so với gửi từng clip. Cùng quy ước với
    ``iter_clips``: video ngắn hơn ``num_frames`` thì lặp frame cuối cho đủ một clip.
    """
    buf: list[np.ndarray] = []
    buf_start = 0  # chỉ số frame của buf[0]
    first = 0  # chỉ số clip đầu tiên của khối đang gom
    total = 0  # số clip đã trả về
    n_frames = 0

    def take(m: int) -> np.ndarray:
        start = first * stride - buf_start
        return np.stack(buf[start : start + (m - 1) * stride + num_frames])

    for idx, frame in enumerate(frames):
        n_frames = idx + 1
        buf.append(frame)
        m = clips_per_chunk if limit is None else min(clips_per_chunk, limit - total)
        # clip cuối của khối (chỉ số first + m - 1) kết thúc ở frame (first + m - 1) * stride + num_frames - 1
        if idx == (first + m - 1) * stride + num_frames - 1:
            yield take(m), m
            first += m
            total += m
            if limit is not None and total >= limit:
                return
            drop = min(first * stride - buf_start, len(buf))
            del buf[:drop]
            buf_start += drop
    if n_frames == 0:
        return
    if n_frames < num_frames:  # video quá ngắn: một clip, lặp frame cuối
        if total == 0:
            yield np.stack(buf + [buf[-1]] * (num_frames - len(buf))), 1
        return
    m = (n_frames - num_frames) // stride + 1 - first  # các clip trọn vẹn còn lại
    if limit is not None:
        m = min(m, limit - total)
    if m > 0:
        yield take(m), m


def iter_batches(
    items: Iterable[np.ndarray], batch_size: int, limit: int | None = None
) -> Iterator[np.ndarray]:
    """Gom ``batch_size`` mảng cùng shape thành một mảng (B, ...); dừng sau ``limit`` phần tử nếu có."""
    batch: list[np.ndarray] = []
    for i, item in enumerate(items):
        if limit is not None and i >= limit:
            break
        batch.append(item)
        if len(batch) == batch_size:
            yield np.stack(batch)
            batch = []
    if batch:
        yield np.stack(batch)


def prefetch(it: Iterable, max_items: int) -> Iterator:
    """Chạy iterator ở thread nền (giải mã video song song với GPU).

    Thread nền dừng khi generator bị đóng (ví dụ ``break`` ở vòng lặp ngoài), không để lại
    tiến trình ffmpeg chạy ngầm.
    """
    q: queue.Queue = queue.Queue(maxsize=max_items)
    sentinel = object()
    stop = threading.Event()

    def put(item) -> bool:
        while not stop.is_set():
            try:
                q.put(item, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def worker():
        try:
            for item in it:
                if not put(item):
                    break
        except BaseException as e:  # chuyển lỗi sang thread chính
            put(e)
        finally:
            close = getattr(it, "close", None)
            if close is not None:
                close()
            put(sentinel)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    try:
        while True:
            item = q.get()
            if item is sentinel:
                return
            if isinstance(item, BaseException):
                raise item
            yield item
    finally:
        stop.set()
        thread.join(timeout=5)


# --------------------------------------------------------------------------- audio


def load_audio(
    path: str,
    sr: int = 16000,
    ffmpeg: str | None = None,
    resampler: str = "librosa",
    res_type: str = "kaiser_best",
) -> np.ndarray | None:
    """Waveform mono float32 ở ``sr`` Hz; None nếu video không có track audio.

    resampler='librosa' (mặc định, sát bản gốc nhất): mô phỏng ``librosa.load(path, sr=16000)`` mà
        ``OnePeaceHubInterface.process_audio`` dùng — ffmpeg giải mã ở sample rate gốc,
        mono = trung bình các kênh, resample bằng librosa.
    resampler='ffmpeg': ffmpeg tự downmix + resample (nhanh hơn, bộ lọc khác).
    ``res_type``: bộ lọc resample của librosa. 'kaiser_best' = mặc định của librosa < 0.10 và là
        lựa chọn khớp nhất với feature DESED của tác giả UniAV (xem README mục 9); 'soxr_hq' =
        mặc định của librosa >= 0.10.
    """
    ffmpeg = ffmpeg or find_ffmpeg()
    base = [ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-i", path, "-vn", "-sn"]
    if resampler == "ffmpeg":
        cmd, native_sr, channels = base + ["-ac", "1", "-ar", str(sr), "-f", "f32le", "pipe:1"], sr, 1
    else:
        native_sr = probe(path, ffmpeg)["sample_rate"]
        if native_sr is None:
            return None
        # giải mã PCM 16-bit, 2 kênh, sample rate gốc — giống wav tách bằng ffmpeg / audioread mà
        # librosa đọc (nguồn mono -> 2 kênh giống nhau, trung bình vẫn là chính nó)
        cmd, channels = base + ["-ac", "2", "-f", "s16le", "pipe:1"], 2
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        msg = proc.stderr.decode("utf-8", "ignore")
        if "does not contain any stream" in msg or "matches no streams" in msg:
            return None
        raise RuntimeError(f"ffmpeg lỗi ({proc.returncode}): {msg[-800:]}")
    if len(proc.stdout) == 0:
        return None
    if resampler == "ffmpeg":
        wav = np.frombuffer(proc.stdout, dtype=np.float32)
    else:
        # = librosa.util.buf_to_float
        wav = np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    wav = wav[: len(wav) // channels * channels].reshape(-1, channels).mean(axis=1)
    if native_sr != sr:
        import librosa

        wav = librosa.resample(wav, orig_sr=native_sr, target_sr=sr, res_type=res_type)
    return np.ascontiguousarray(wav, dtype=np.float32)


def load_audio_file(path: str, sr: int = 16000, res_type: str = "kaiser_best") -> np.ndarray | None:
    """Đọc file audio riêng (wav/flac/...) như ``OnePeaceHubInterface.process_audio``.

    ``librosa.load(path, sr=16000, res_type=...)`` — mono = trung bình kênh, resample nếu sample
    rate khác 16 kHz. None nếu file rỗng. Xem ``load_audio`` về ý nghĩa ``res_type``.
    """
    import librosa

    wav, _ = librosa.load(path, sr=sr, res_type=res_type)
    return np.ascontiguousarray(wav, dtype=np.float32) if wav.size else None


# --------------------------------------------------------------------------- tiện ích chung


def num_windows(total: int, window: int, hop: int) -> int:
    """Số cửa sổ trượt (luôn >= 1, giống cách xử lý video/audio ngắn)."""
    return max(0, (total - window) // hop) + 1


def list_video_ids(
    video_dir: str,
    ids_from: str | None = None,
    ext: str = ".mp4",
    must_exist: bool = True,
    file_prefix: str = "",
) -> list[str]:
    """Danh sách video id cần xử lý.

    - không có ids_from: mọi file ``*ext`` trong video_dir;
    - ids_from là file .json: annotation dạng UniAV (``{"database": {vid: ...}}``) hoặc
      youcookii_*_preprocess.json (``{"database": {seg_id: {"video_id": ...}}}``);
    - ids_from là file .txt: mỗi dòng một id.
    must_exist=True: chỉ giữ các id có file ``<file_prefix><id><ext>`` trong video_dir.
    ``file_prefix``: tiền tố của tên file mà id không có (DESED validation: file ``Y<id>.wav``).
    """
    if ids_from is None:
        ids = sorted(
            os.path.splitext(f)[0][len(file_prefix) :]
            for f in os.listdir(video_dir)
            if f.endswith(ext) and f.startswith(file_prefix)
        )
    elif ids_from.endswith(".json"):
        with open(ids_from, encoding="utf-8") as f:
            db = json.load(f)["database"]
        ids = sorted({v.get("video_id", k) if isinstance(v, dict) else k for k, v in db.items()})
    else:
        with open(ids_from, encoding="utf-8") as f:
            ids = sorted({line.strip() for line in f if line.strip()})
    if not must_exist:
        return ids
    return [i for i in ids if os.path.isfile(os.path.join(video_dir, file_prefix + i + ext))]


def shard(items: list[str], num_shards: int, shard_id: int) -> list[str]:
    """Phần thứ ``shard_id`` khi chia ``items`` xen kẽ thành ``num_shards`` phần."""
    if not 0 <= shard_id < num_shards:
        raise ValueError(f"shard_id phải nằm trong [0, {num_shards}), nhận {shard_id}")
    return items[shard_id::num_shards]


def save_npy_atomic(path: str, arr: np.ndarray) -> None:
    """Ghi file tạm rồi rename -> không để lại file .npy hỏng nếu Colab bị ngắt giữa chừng."""
    tmp = path + ".tmp.npy"
    np.save(tmp, arr)
    os.replace(tmp, path)


# --------------------------------------------------------------------------- resume


def load_done_ids(output_dir: str, suffix: str, done_prefix: str, extra: Iterable[str] = ()) -> set[str]:
    """ID đã trích xuất xong, gộp từ:

    - file ``<id><suffix>`` trong ``output_dir`` (một lần ``listdir``, nhanh hơn hỏi từng file trên Drive);
    - các file ``<done_prefix>*.txt`` trong ``output_dir`` (script tự ghi thêm mỗi khi xong một video);
    - ``extra``: file .txt (mỗi dòng một id, cột đầu nếu có tab) hoặc thư mục chứa ``<id><suffix>``
      (ví dụ output của phiên Kaggle trước, gắn vào làm input).
    """
    done: set[str] = set()

    def from_dir(d: str) -> None:
        for f in os.listdir(d):
            if f.endswith(suffix) and not f.endswith(".tmp.npy"):
                done.add(f[: -len(suffix)])
            elif f.startswith(done_prefix) and f.endswith(".txt"):
                from_txt(os.path.join(d, f))

    def from_txt(p: str) -> None:
        with open(p, encoding="utf-8") as fh:
            done.update(line.split("\t")[0].strip() for line in fh if line.strip())

    if os.path.isdir(output_dir):
        from_dir(output_dir)
    for p in extra:
        if os.path.isdir(p):
            from_dir(p)
        elif os.path.isfile(p):
            from_txt(p)
        else:
            print(f"[warn] --done_list: không tìm thấy {p}", flush=True)
    return done


def append_line(path: str, line: str) -> None:
    """Ghi thêm một dòng (mở/đóng mỗi lần để dòng đã ghi không mất khi phiên bị tắt đột ngột)."""
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")
        f.flush()
        os.fsync(f.fileno())


# --------------------------------------------------------------------------- song song


_WORKER_DONE = "__worker_done__"


def _parallel_worker(fn, task_q, out_q) -> None:
    while True:
        item = task_q.get()
        if item is None:
            break
        try:
            for out in fn(item):
                out_q.put(out)
        except BaseException as e:  # fn phải tự báo lỗi từng item; tới đây là lỗi ngoài dự kiến
            out_q.put(("__error__", item, repr(e)))
    out_q.put(_WORKER_DONE)


def iter_parallel(fn, items: list, workers: int, max_queue: int) -> Iterator:
    """Chạy generator ``fn(item)`` cho từng item trên ``workers`` tiến trình, trả kết quả theo thứ tự xong.

    Mỗi item do đúng một tiến trình xử lý, nên các output của cùng một item vẫn đúng thứ tự.
    ``max_queue`` giới hạn số output chờ trong hàng đợi (chặn RAM khi GPU chậm hơn khâu giải mã).
    Tiến trình riêng (không phải thread) để giải mã/resize không tranh GIL với vòng lặp GPU.
    ``fn`` phải pickle được (hàm ở cấp module, hoặc functools.partial của nó).
    """
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    task_q, out_q = ctx.Queue(), ctx.Queue(maxsize=max_queue)
    for item in items:
        task_q.put(item)
    for _ in range(workers):
        task_q.put(None)
    procs = [
        ctx.Process(target=_parallel_worker, args=(fn, task_q, out_q), daemon=True) for _ in range(workers)
    ]
    for p in procs:
        p.start()
    try:
        alive = workers
        while alive:
            out = out_q.get()
            if isinstance(out, str) and out == _WORKER_DONE:
                alive -= 1
                continue
            yield out
    finally:
        for p in procs:
            if p.is_alive():
                p.terminate()
        for p in procs:
            p.join(timeout=5)


def resolve_gpus(spec: str) -> list[int]:
    """'auto' = mọi GPU thấy được; '0,1' = danh sách; '' / 'cpu' / không có GPU = []."""
    import torch

    if not torch.cuda.is_available() or spec in ("", "cpu"):
        return []
    if spec == "auto":
        return list(range(torch.cuda.device_count()))
    return [int(g) for g in spec.split(",") if g.strip()]


def auto_workers(n_gpus: int, per_gpu_max: int = 3) -> int:
    """Số tiến trình giải mã cho mỗi GPU: chia đều CPU (ffmpeg tự dùng nhiều luồng nên chỉ lấy một nửa)."""
    cpus = os.cpu_count() or 2
    return max(1, min(per_gpu_max, cpus // (2 * max(n_gpus, 1))))


def run_on_gpus(target, todo: list, gpus: list[int], *extra) -> None:
    """Chia ``todo`` xen kẽ cho từng GPU, mỗi GPU một tiến trình: ``target(todo_con, device, rank, *extra)``.

    Một GPU (hoặc CPU) thì chạy thẳng trong tiến trình hiện tại. Lỗi ở tiến trình con -> thoát mã != 0.
    """
    if len(gpus) <= 1:
        target(todo, f"cuda:{gpus[0]}" if gpus else "cpu", 0, *extra)
        return
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    lock = ctx.Lock()  # nạp checkpoint lần lượt từng GPU, tránh 2 bản 6 GB cùng lúc trong RAM
    procs = []
    for rank, g in enumerate(gpus):
        p = ctx.Process(target=target, args=(todo[rank :: len(gpus)], f"cuda:{g}", rank, *extra, lock))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
    bad = [p.exitcode for p in procs if p.exitcode != 0]
    if bad:
        raise SystemExit(f"{len(bad)} tiến trình GPU lỗi (exit code {bad})")
