"""Chuyển metadata YouCookII (youcookii_{train,val}_preprocess.json) sang annotation dạng UniAV.

Định dạng UniAV (giống data/desed/dcase_all.json, đọc bởi libs/datasets/anet.py):
{
  "version": "YouCookII",
  "database": {
    "<video_id>": {
      "subset": "training" | "validation",
      "duration": 241.62,
      "annotations": [{"segment": [90.0, 102.0], "label": "...", "label_id": 0, "sentence": "..."}]
    }
  }
}

YouCookII chỉ có mô tả câu cho từng bước, KHÔNG có nhãn lớp, nên cần chọn cách gán nhãn:
  --label_mode step   : một lớp duy nhất "cooking step" (localization không phân lớp, num_classes = 1)
  --label_mode recipe : lớp = loại món (recipe_type, 89 lớp). Tên món lấy từ --recipe_names
                        (label_foodtype.csv của YouCookII: "id,tên"), nếu không có thì "recipe <id>".

  python build_youcookii_annotations.py \
      --train_json D:/Học/KL/Data/YouCookII/metadata/youcookii_train_preprocess.json \
      --val_json   D:/Học/KL/Data/YouCookII/metadata/youcookii_val_preprocess.json \
      --output youcookii_all.json --video_dir D:/Học/KL/Data/YouCookII/videos
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, OrderedDict

import numpy as np


def load_recipe_names(path: str | None) -> dict[str, str]:
    """Đọc label_foodtype.csv ('id,tên') -> {id: tên}; rỗng nếu không có file."""
    names = {}
    if path:
        with open(path, encoding="utf-8") as f:
            for line in f:
                parts = [p.strip() for p in line.strip().split(",", 1)]
                if len(parts) == 2 and parts[0].isdigit():
                    names[parts[0]] = parts[1]
    return names


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train_json", required=True)
    p.add_argument("--val_json", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--label_mode", choices=["step", "recipe"], default="step")
    p.add_argument("--recipe_names", default=None, help="label_foodtype.csv (tuỳ chọn)")
    p.add_argument("--video_dir", default=None, help="bỏ các video không có file .mp4")
    p.add_argument(
        "--feat_dir",
        default=None,
        help="bỏ các video chưa có đủ 2 file feature *_one_peace_{video_finetune,audio}.npy",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    recipe_names = load_recipe_names(args.recipe_names)
    database = OrderedDict()
    stats = Counter()

    for subset, path in (("training", args.train_json), ("validation", args.val_json)):
        with open(path, encoding="utf-8") as f:
            segs = json.load(f)["database"]
        # sắp theo video rồi theo thứ tự bước (seg id dạng <video_id>_<k>)
        items = sorted(segs.items(), key=lambda kv: (kv[1]["video_id"], int(kv[0].rsplit("_", 1)[-1])))
        for _seg_id, s in items:
            vid = s["video_id"]
            if args.video_dir and not os.path.isfile(os.path.join(args.video_dir, vid + ".mp4")):
                stats[f"{subset}: bỏ đoạn (không có video)"] += 1
                continue
            if args.feat_dir and not all(
                os.path.isfile(os.path.join(args.feat_dir, f"{vid}_one_peace_{m}.npy"))
                for m in ("video_finetune", "audio")
            ):
                stats[f"{subset}: bỏ đoạn (chưa có feature)"] += 1
                continue

            duration = float(s["duration"])
            start, end = float(s["segment"][0]), min(float(s["segment"][1]), duration)
            if end <= start:  # anet.py chia cho (end - start) -> đoạn rỗng gây lỗi
                stats[f"{subset}: bỏ đoạn (độ dài <= 0)"] += 1
                continue

            entry = database.setdefault(vid, {"subset": subset, "duration": duration, "annotations": []})
            assert entry["subset"] == subset, f"{vid} xuất hiện ở cả train và val"
            if args.label_mode == "step":
                label = "cooking step"
            else:
                rid = str(s["recipe_type"])
                label = recipe_names.get(rid, f"recipe {rid}")
            entry["annotations"].append(
                {"segment": [start, end], "label": label, "sentence": s.get("sentence", "")}
            )
            stats[f"{subset}: đoạn"] += 1

    # label_id theo thứ tự tên nhãn (ổn định giữa các lần chạy)
    labels = sorted({a["label"] for v in database.values() for a in v["annotations"]})
    label_to_id = {name: i for i, name in enumerate(labels)}
    for v in database.values():
        for a in v["annotations"]:
            a["label_id"] = label_to_id[a["label"]]

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(
            {"version": "YouCookII", "label_mode": args.label_mode, "database": database},
            f,
            ensure_ascii=False,
            indent=1,
        )

    for subset in ("training", "validation"):
        vids = [v for v in database.values() if v["subset"] == subset]
        n_seg = [len(v["annotations"]) for v in vids]
        lens = [a["segment"][1] - a["segment"][0] for v in vids for a in v["annotations"]]
        durs = [v["duration"] for v in vids]
        if vids:
            print(
                f"{subset}: {len(vids)} video, {sum(n_seg)} đoạn "
                f"(TB {np.mean(n_seg):.1f}/video), video TB {np.mean(durs):.0f}s, "
                f"đoạn TB {np.mean(lens):.1f}s (min {min(lens):.1f}, max {max(lens):.1f})"
            )
    for k, v in sorted(stats.items()):
        if "bỏ" in k:
            print(f"  {k}: {v}")
    print(f"{len(labels)} nhãn -> num_classes = {len(labels)}. Đã ghi {args.output}")


if __name__ == "__main__":
    main()
