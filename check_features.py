"""Kiểm tra thư mục feature trước khi đưa vào UniAV.

Với mỗi video trong annotation: đủ 2 file chưa, shape (T, 1536), số bước visual/audio có lệch
nhiều không (UniAV cắt về min), T có khớp thời lượng không, chuẩn L2 (audio ~1, visual ~20-30).

  python check_features.py --anno annotations/youcookii_all.json --feat_dir /content/feats/youcookii --stride 8
"""

import argparse
import json
import os

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--anno", required=True)
    p.add_argument("--feat_dir", required=True)
    p.add_argument("--stride", type=int, default=8, help="bước (frame @16fps) đã dùng khi trích xuất")
    p.add_argument("--num_frames", type=int, default=16)
    p.add_argument("--fps", type=int, default=16)
    args = p.parse_args()

    with open(args.anno, "r", encoding="utf-8") as f:
        db = json.load(f)["database"]

    missing, bad, rows = [], [], []
    for vid, v in db.items():
        fv = os.path.join(args.feat_dir, f"{vid}_one_peace_video_finetune.npy")
        fa = os.path.join(args.feat_dir, f"{vid}_one_peace_audio.npy")
        if not (os.path.isfile(fv) and os.path.isfile(fa)):
            missing.append(vid)
            continue
        vis, aud = np.load(fv, mmap_mode="r"), np.load(fa, mmap_mode="r")
        if vis.ndim != 2 or aud.ndim != 2 or vis.shape[1] != 1536 or aud.shape[1] != 1536:
            bad.append((vid, vis.shape, aud.shape))
            continue
        expected = max(0, int((v["duration"] * args.fps - args.num_frames) // args.stride)) + 1
        rows.append((vid, vis.shape[0], aud.shape[0], expected,
                     float(np.linalg.norm(vis[:50], axis=1).mean()), float(np.linalg.norm(aud[:50], axis=1).mean())))

    print(f"{len(db)} video trong annotation, {len(rows)} hợp lệ, thiếu file: {len(missing)}, sai shape: {len(bad)}")
    if missing:
        print("  ví dụ thiếu:", missing[:10])
    if bad:
        print("  ví dụ sai shape:", bad[:5])
    if not rows:
        return
    arr = np.array([r[1:] for r in rows], dtype=np.float64)
    diff_va = np.abs(arr[:, 0] - arr[:, 1])
    diff_dur = np.abs(np.minimum(arr[:, 0], arr[:, 1]) - arr[:, 2]) * args.stride / args.fps
    print(f"|T_visual - T_audio|: TB {diff_va.mean():.2f}, max {diff_va.max():.0f} bước")
    print(f"lệch so với thời lượng trong annotation: TB {diff_dur.mean():.2f}s, max {diff_dur.max():.1f}s")
    print(f"chuẩn L2 visual TB {arr[:, 3].mean():.2f} (DESED của tác giả ~20-30), audio TB {arr[:, 4].mean():.3f} (~1.0)")
    worst = np.argsort(-diff_dur)[:5]
    print("video lệch thời lượng nhiều nhất:", [(rows[i][0], f"{diff_dur[i]:.1f}s") for i in worst])


if __name__ == "__main__":
    main()
