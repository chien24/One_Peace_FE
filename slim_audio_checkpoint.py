"""Tách nhánh AUDIO từ checkpoint ONE-PEACE (one-peace.pt, 15.5 GB, 3.89B tham số).

Lý do: fairseq nạp nguyên checkpoint vào RAM rồi mới dựng model -> không chạy nổi trên
Colab thường (RAM 12.7 GB) hay máy 16 GB. Script này giữ đúng các tham số ONE-PEACE dùng khi
head_type='audio' (bỏ khoá chứa 'text_' / 'image_', giống hệt
OnePeaceRetrievalModel.remove_pretraining_modules). Mặc định giữ nguyên fp32 (~5.7 GB, 1.53B tham số);
--fp16 cho file ~2.9 GB.

Checkpoint được đọc trực tiếp từ file zip của torch.save, từng tensor một (không dùng
torch.load / mmap), nên RAM cần ~ kích thước file đầu ra và không cần cài fairseq.
(torch.load(mmap=True) bị lỗi access violation trên Windows vì mmap copy-on-write 15.5 GB
bị tính hết vào bộ nhớ commit.)

  python slim_audio_checkpoint.py --input one-peace.pt --output one-peace-audio.pt
"""

import argparse
import os
import pickle
import zipfile

import numpy as np
import torch

_STORAGE_DTYPES = {
    "FloatStorage": torch.float32, "HalfStorage": torch.float16, "BFloat16Storage": torch.bfloat16,
    "DoubleStorage": torch.float64, "LongStorage": torch.int64, "IntStorage": torch.int32,
    "ShortStorage": torch.int16, "CharStorage": torch.int8, "ByteStorage": torch.uint8,
    "BoolStorage": torch.bool,
}


class _LazyTensor:
    def __init__(self, storage, offset, size, stride):
        self.storage, self.offset, self.size, self.stride = storage, offset, tuple(size), tuple(stride)

    def numel(self):
        return int(np.prod(self.size)) if self.size else 1


class _Unpickler(pickle.Unpickler):
    """Unpickle data.pkl, thay tensor bằng _LazyTensor (chưa đọc dữ liệu)."""

    def find_class(self, module, name):
        if module == "torch" and name in _STORAGE_DTYPES:
            return _STORAGE_DTYPES[name]
        if module == "torch._utils" and name == "_rebuild_tensor_v2":
            return lambda storage, offset, size, stride, *args, **kw: _LazyTensor(storage, offset, size, stride)
        return super().find_class(module, name)

    def persistent_load(self, saved_id):
        # ('storage', storage_type, key, location, numel)
        _, storage_type, key, _, numel = saved_id
        dtype = storage_type if isinstance(storage_type, torch.dtype) else storage_type.dtype
        return (key, dtype, numel)


def read_tensor(zf, prefix, lt: _LazyTensor) -> torch.Tensor:
    key, dtype, numel = lt.storage
    raw = zf.read(f"{prefix}/data/{key}")
    storage = torch.frombuffer(bytearray(raw), dtype=dtype) if numel else torch.empty(0, dtype=dtype)
    return torch.as_strided(storage, lt.size, lt.stride, lt.offset).clone()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--fp16", action="store_true", help="lưu fp16 cho nhẹ (mặc định giữ fp32 như checkpoint gốc)")
    p.add_argument("--onepeace_repo", default=None, help="(không còn cần, giữ để tương thích)")
    args = p.parse_args()

    with zipfile.ZipFile(args.input) as zf:
        pkl_name = next(n for n in zf.namelist() if n.endswith("data.pkl"))
        prefix = pkl_name[:-len("/data.pkl")]
        with zf.open(pkl_name) as f:
            state = _Unpickler(f).load()
        print("Các khoá cấp cao:", list(state.keys()))
        print("task:", state["cfg"]["task"].get("_name"), "| head_type:", state["cfg"]["task"].get("head_type"),
              "| model:", state["cfg"]["model"].get("_name"))

        kept, dropped, n_params = {}, 0, 0
        for k, lt in state["model"].items():
            if "text_" in k or "image_" in k:
                dropped += 1
                continue
            t = read_tensor(zf, prefix, lt) if isinstance(lt, _LazyTensor) else lt
            if torch.is_floating_point(t) and args.fp16:
                t = t.half()
            kept[k] = t
            n_params += t.numel()
    print(f"Giữ {len(kept)} tensor ({n_params / 1e9:.2f}B tham số), bỏ {dropped} tensor text/image")

    slim = {k: state[k] for k in ("args", "cfg", "optimizer_history", "task_state") if k in state}
    slim["model"] = kept
    torch.save(slim, args.output)
    print(f"Đã lưu {args.output} ({os.path.getsize(args.output) / 1024 ** 3:.2f} GB)")


if __name__ == "__main__":
    main()
