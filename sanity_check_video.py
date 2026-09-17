"""Kiểm tra nhanh backbone video + tiền xử lý trên vài clip của 1 video.

In ra: top-5 nhãn Kinetics-400 (nếu đúng tiền xử lý, video nấu ăn phải ra các lớp như
"cooking egg", "making a sandwich", "cutting vegetables"...), chuẩn L2 của đặc trưng
(feature video của tác giả trên DESED có chuẩn ~20-30), và sai khác SDPA vs bmm.

  python sanity_check_video.py --checkpoint onepeace_video_k400.pth --video path/to/video.mp4
"""

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F

from onepeace_video_backbone import IMG_MEAN, IMG_STD, load_k400_checkpoint
from video_io import iter_clips, iter_video_frames

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--video", required=True)
    p.add_argument("--start_sec", type=float, default=0.0, help="bỏ qua N giây đầu")
    p.add_argument("--num_clips", type=int, default=2)
    p.add_argument("--stride", type=int, default=16)
    p.add_argument("--label_map", default=os.path.join(HERE, "label_map_k400.txt"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--resize_backend", choices=["cv2", "ffmpeg"], default="cv2")
    p.add_argument("--compare_bmm", action="store_true", help="so sánh SDPA với attention gốc")
    p.add_argument("--compare_fp16", action="store_true", help="(GPU) so feature fp32 với fp16")
    args = p.parse_args()

    labels = open(args.label_map, encoding="utf-8").read().splitlines()
    model, head = load_k400_checkpoint(args.checkpoint, with_head=True)
    dtype = torch.float32  # fp32 = cách sát bản gốc nhất
    if args.device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    model, head = model.to(args.device, dtype), head.to(args.device, dtype)

    skip = int(args.start_sec * 16)
    frames = (f for i, f in enumerate(iter_video_frames(args.video, resize_backend=args.resize_backend)) if i >= skip)
    clips = []
    for c in iter_clips(frames, 16, args.stride):
        clips.append(c)
        if len(clips) == args.num_clips:
            break

    x = torch.from_numpy(np.stack(clips)).permute(0, 4, 1, 2, 3).float()
    x = (x - torch.tensor(IMG_MEAN).view(1, 3, 1, 1, 1)) / torch.tensor(IMG_STD).view(1, 3, 1, 1, 1)
    x = x.to(args.device, dtype)
    with torch.inference_mode():
        feats = model.extract_clip_features(x)
        probs = head(feats).float().softmax(-1)
        for i in range(len(clips)):
            top = probs[i].topk(5)
            t = args.start_sec + i * args.stride / 16
            print(f"clip @{t:.1f}s  |feat|={feats[i].float().norm():.2f}  " +
                  ", ".join(f"{labels[j]} ({v:.2f})" for v, j in zip(top.values.tolist(), top.indices.tolist())))
        if args.compare_bmm:
            for m in model.modules():
                if hasattr(m, "use_sdpa"):
                    m.use_sdpa = False
            feats_bmm = model.extract_clip_features(x)
            rel = ((feats - feats_bmm).float().norm() / feats_bmm.float().norm()).item()
            print(f"sai khác tương đối SDPA vs bmm: {rel:.2e}")
            for m in model.modules():
                if hasattr(m, "use_sdpa"):
                    m.use_sdpa = True
        if args.compare_fp16 and args.device.startswith("cuda"):
            model.half()
            feats16 = model.extract_clip_features(x.half()).float()
            cos = F.cosine_similarity(feats16, feats.float(), dim=-1)
            rel = ((feats16 - feats.float()).norm() / feats.float().norm()).item()
            print(f"fp16 vs fp32: cos min {cos.min().item():.6f}, sai khác tương đối {rel:.2e}")


if __name__ == "__main__":
    main()
