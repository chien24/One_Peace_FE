"""Đọc video/audio bằng ffmpeg (stream, không bung toàn bộ video vào RAM) và các tiện ích chung."""

import json
import os
import queue
import re
import shutil
import subprocess
import tempfile
import threading
from typing import Iterator, List, Optional

import numpy as np


def find_ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        raise RuntimeError("Không tìm thấy ffmpeg. Cài bằng `apt install ffmpeg` hoặc `pip install imageio-ffmpeg`.")


def probe(path: str, ffmpeg: Optional[str] = None) -> dict:
    """Thông tin cơ bản từ `ffmpeg -i` (không cần ffprobe): duration, width, height, sample_rate."""
    ffmpeg = ffmpeg or find_ffmpeg()
    proc = subprocess.run([ffmpeg, "-hide_banner", "-noautorotate", "-i", path], capture_output=True,
                          text=True, encoding="utf-8", errors="ignore")
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


def probe_duration(path: str, ffmpeg: Optional[str] = None) -> Optional[float]:
    return probe(path, ffmpeg)["duration"]


def _run_ffmpeg_stream(cmd: List[str], chunk_bytes: int) -> Iterator[bytes]:
    """Chạy ffmpeg, trả về từng khối `chunk_bytes` byte từ stdout. Lỗi ffmpeg -> RuntimeError."""
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


def mmaction_resize_center_crop(img: np.ndarray, size: int = 256) -> np.ndarray:
    """Đúng pipeline test của ONE-PEACE K400 (mmaction2):
    Resize(scale=(-1, size)) = mmcv.rescale_size + mmcv.imresize(cv2 INTER_LINEAR), rồi CenterCrop(size)."""
    import cv2
    h, w = img.shape[:2]
    scale = size / min(h, w)                                   # rescale_size((w, h), (inf, size))
    new_w, new_h = int(w * scale + 0.5), int(h * scale + 0.5)
    if (new_w, new_h) != (w, h):
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    left, top = (new_w - size) // 2, (new_h - size) // 2       # CenterCrop
    return np.ascontiguousarray(img[top:top + size, left:left + size])


def iter_video_frames(path: str, fps: int = 16, size: int = 256, ffmpeg: Optional[str] = None,
                      resize_backend: str = "cv2") -> Iterator[np.ndarray]:
    """Frame RGB uint8 (size, size, 3): lấy mẫu lại `fps` -> resize cạnh ngắn = size -> center crop.

    resize_backend='cv2' (mặc định, sát bản gốc nhất): ffmpeg chỉ giải mã + đổi fps ở độ phân giải gốc,
        resize/crop làm bằng cv2 y như mmaction2 (pipeline ONE-PEACE fine-tune K400).
    resize_backend='ffmpeg': resize/crop luôn trong ffmpeg (nhanh hơn, nội suy hơi khác).
    Không tự xoay video theo metadata (-noautorotate), giống decord mà mmaction2 dùng.
    """
    ffmpeg = ffmpeg or find_ffmpeg()
    base = [ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-noautorotate", "-i", path, "-an", "-sn"]
    if resize_backend == "ffmpeg":
        vf = (f"fps={fps},"
              f"scale='if(lte(iw,ih),{size},-2)':'if(lte(iw,ih),-2,{size})':flags=bilinear,"
              f"crop={size}:{size}")
        cmd = base + ["-vf", vf, "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1"]
        for buf in _run_ffmpeg_stream(cmd, size * size * 3):
            yield np.frombuffer(buf, dtype=np.uint8).reshape(size, size, 3)
        return

    info = probe(path, ffmpeg)
    w, h = info["width"], info["height"]
    if not w or not h:
        raise RuntimeError(f"không đọc được kích thước video: {path}")
    cmd = base + ["-vf", f"fps={fps}", "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1"]
    for buf in _run_ffmpeg_stream(cmd, w * h * 3):
        yield mmaction_resize_center_crop(np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3), size)


def iter_clips(frames: Iterator[np.ndarray], num_frames: int = 16, stride: int = 4) -> Iterator[np.ndarray]:
    """Cửa sổ trượt `num_frames` frame, bước `stride` -> mảng (num_frames, H, W, 3).

    Số clip = floor((N - num_frames) / stride) + 1, clip i bắt đầu ở frame i * stride.
    Video ngắn hơn num_frames: lặp lại frame cuối để đủ 1 clip.
    """
    buf: List[np.ndarray] = []
    next_start = 0   # chỉ số frame bắt đầu của clip kế tiếp
    buf_start = 0    # chỉ số frame của buf[0]
    emitted = False
    for idx, frame in enumerate(frames):
        buf.append(frame)
        if idx == next_start + num_frames - 1:
            offset = next_start - buf_start
            yield np.stack(buf[offset:offset + num_frames])
            emitted = True
            next_start += stride
            removed = min(next_start - buf_start, len(buf))
            del buf[:removed]
            buf_start += removed
    if not emitted and buf:
        clip = buf + [buf[-1]] * (num_frames - len(buf))
        yield np.stack(clip[:num_frames])


def prefetch(it: Iterator, max_items: int) -> Iterator:
    """Chạy iterator ở thread nền (giải mã video song song với GPU)."""
    q: "queue.Queue" = queue.Queue(maxsize=max_items)
    sentinel = object()

    def worker():
        try:
            for item in it:
                q.put(item)
        except BaseException as e:  # chuyển lỗi sang thread chính
            q.put(e)
        finally:
            q.put(sentinel)

    threading.Thread(target=worker, daemon=True).start()
    while True:
        item = q.get()
        if item is sentinel:
            return
        if isinstance(item, BaseException):
            raise item
        yield item


def load_audio(path: str, sr: int = 16000, ffmpeg: Optional[str] = None,
               resampler: str = "librosa") -> Optional[np.ndarray]:
    """Waveform mono float32 ở `sr` Hz; None nếu video không có track audio.

    resampler='librosa' (mặc định, sát bản gốc nhất): mô phỏng librosa.load(path, sr=16000) mà
        OnePeaceHubInterface.process_audio dùng — ffmpeg giải mã ở sample rate gốc,
        mono = trung bình các kênh, resample bằng librosa (res_type mặc định 'soxr_hq').
    resampler='ffmpeg': ffmpeg tự downmix + resample (nhanh hơn, bộ lọc khác).
    """
    ffmpeg = ffmpeg or find_ffmpeg()
    base = [ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-i", path, "-vn", "-sn"]
    if resampler == "ffmpeg":
        cmd, native_sr, channels = base + ["-ac", "1", "-ar", str(sr), "-f", "f32le", "pipe:1"], sr, 1
    else:
        native_sr = probe(path, ffmpeg)["sample_rate"]
        if native_sr is None:
            return None
        # giải mã PCM 16-bit, 2 kênh, sample rate gốc — giống wav tách bằng ffmpeg / audioread mà librosa đọc
        # (nguồn mono -> 2 kênh giống nhau, trung bình vẫn là chính nó)
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
        wav = np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0  # librosa.util.buf_to_float
    wav = wav[: len(wav) // channels * channels].reshape(-1, channels).mean(axis=1)
    if native_sr != sr:
        import librosa
        wav = librosa.resample(wav, orig_sr=native_sr, target_sr=sr, res_type="soxr_hq")  # = librosa.load 0.10
    return np.ascontiguousarray(wav, dtype=np.float32)


def num_windows(total: int, window: int, hop: int) -> int:
    return max(0, (total - window) // hop) + 1


def list_video_ids(video_dir: str, ids_from: Optional[str] = None, ext: str = ".mp4") -> List[str]:
    """Danh sách video id cần xử lý.

    - không có ids_from: mọi file *ext trong video_dir;
    - ids_from là file .json: annotation dạng UniAV ({"database": {vid: ...}}) hoặc
      youcookii_*_preprocess.json ({"database": {seg_id: {"video_id": ...}}});
    - ids_from là file .txt: mỗi dòng một id.
    Chỉ giữ các id có file video tồn tại.
    """
    if ids_from is None:
        ids = sorted(os.path.splitext(f)[0] for f in os.listdir(video_dir) if f.endswith(ext))
    elif ids_from.endswith(".json"):
        with open(ids_from, "r", encoding="utf-8") as f:
            db = json.load(f)["database"]
        ids = sorted({v.get("video_id", k) if isinstance(v, dict) else k for k, v in db.items()})
    else:
        with open(ids_from, "r", encoding="utf-8") as f:
            ids = sorted({line.strip() for line in f if line.strip()})
    return [i for i in ids if os.path.isfile(os.path.join(video_dir, i + ext))]


def shard(items: List[str], num_shards: int, shard_id: int) -> List[str]:
    assert 0 <= shard_id < num_shards
    return items[shard_id::num_shards]


def save_npy_atomic(path: str, arr: np.ndarray) -> None:
    """Ghi file tạm rồi rename -> không để lại file .npy hỏng nếu Colab bị ngắt giữa chừng."""
    tmp = path + ".tmp.npy"
    np.save(tmp, arr)
    os.replace(tmp, path)
