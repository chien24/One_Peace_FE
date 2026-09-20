"""Kiểm tra nhanh backbone video + tiền xử lý trên vài clip của 1 video.

In ra: top-5 nhãn Kinetics-400 (nếu đúng tiền xử lý, video nấu ăn phải ra các lớp như
"cooking egg", "making a sandwich", "cutting vegetables"...), chuẩn L2 của đặc trưng
(feature video của tác giả trên DESED có chuẩn ~20-30), sai khác SDPA vs matmul và fp16/bf16 vs fp32.

  python sanity_check_video.py --checkpoint onepeace_video_k400.pth --video path/to/video.mp4
"""

from __future__ import annotations

import argparse
import itertools
import os

import numpy as np
import torch
import torch.nn.functional as F

from onepeace_video_backbone import IMG_MEAN, IMG_STD, load_k400_checkpoint
from video_io import iter_clips, iter_video_frames

HERE = os.path.dirname(os.path.abspath(__file__))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--video", required=True)
    p.add_argument("--start_sec", type=float, default=0.0, help="bỏ qua N giây đầu")
    p.add_argument("--num_clips", type=int, default=2)
    p.add_argument("--stride", type=int, default=16)
    p.add_argument("--label_map", default=os.path.join(HERE, "label_map_k400.txt"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--resize_backend", choices=["cv2", "ffmpeg"], default="cv2")
    p.add_argument("--compare_bmm", action="store_true", help="so sánh SDPA với attention matmul gốc")
    p.add_argument("--compare_fp16", action="store_true", help="(GPU) so feature fp16 và bf16 với fp32")
    return p.parse_args()


def describe_diff(name: str, feats: torch.Tensor, ref: torch.Tensor) -> str:
    feats, ref = feats.float(), ref.float()
    cos = F.cosine_similarity(feats, ref, dim=-1)
    rel = ((feats - ref).norm() / ref.norm()).item()
    return f"{name}: cos min {cos.min().item():.6f}, sai khác tương đối {rel:.2e}"


def main() -> None:
    args = parse_args()
    with open(args.label_map, encoding="utf-8") as f:
        labels = f.read().splitlines()

    model, head = load_k400_checkpoint(args.checkpoint, with_head=True)
    if args.device.startswith("cuda"):
        # fp32 = cách sát bản gốc nhất -> tắt TF32
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    model, head = model.to(args.device), head.to(args.device)

    skip = int(args.start_sec * 16)
    frames = itertools.islice(iter_video_frames(args.video, resize_backend=args.resize_backend), skip, None)
    clips = list(itertools.islice(iter_clips(frames, 16, args.stride), args.num_clips))

    x = torch.from_numpy(np.stack(clips)).permute(0, 4, 1, 2, 3).float()
    x = (x - torch.tensor(IMG_MEAN).view(1, 3, 1, 1, 1)) / torch.tensor(IMG_STD).view(1, 3, 1, 1, 1)
    x = x.to(args.device)

    with torch.inference_mode():
        feats = model.extract_clip_features(x)
        probs = head(feats).softmax(-1)
        for i in range(len(clips)):
            top = probs[i].topk(5)
            t = args.start_sec + i * args.stride / 16
            preds = ", ".join(
                f"{labels[j]} ({v:.2f})"
                for v, j in zip(top.values.tolist(), top.indices.tolist(), strict=True)
            )
            print(f"clip @{t:.1f}s  |feat|={feats[i].norm():.2f}  {preds}")

        if args.compare_bmm:
            model.set_use_sdpa(False)
            print(describe_diff("SDPA vs matmul", feats, model.extract_clip_features(x)))
            model.set_use_sdpa(True)

        if args.compare_fp16 and args.device.startswith("cuda"):
            for name, dtype in (("fp16 vs fp32", torch.float16), ("bf16 vs fp32", torch.bfloat16)):
                # chuyển từ trọng số fp32 gốc (checkpoint mmap) để tránh làm tròn hai lần
                low = load_k400_checkpoint(args.checkpoint).to(args.device, dtype)
                print(describe_diff(name, low.extract_clip_features(x.to(dtype)), feats))
                del low
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
