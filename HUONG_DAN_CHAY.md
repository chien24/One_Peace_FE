# Hướng dẫn chạy trích xuất feature ONE-PEACE cho YouCookII

Tài liệu này chỉ nói **cách chạy**. Lý do chọn từng cách xử lý và các số liệu kiểm chứng nằm trong [README.md](README.md).

Kết quả cuối cùng: mỗi video có 2 file trong thư mục feature, đúng định dạng UniAV đọc:
```
<video_id>_one_peace_video_finetune.npy   (T, 1536)
<video_id>_one_peace_audio.npy            (T, 1536)
```

---

## 0. Chuẩn bị (làm 1 lần)

### 0.1 Chuẩn bị trên Google Drive

| Thứ | Ở đâu | Ghi chú |
|---|---|---|
| Code `One_Peace` | clone từ GitHub (vào Drive hoặc `/content`) | `.gitignore` đã loại checkpoint, nên repo không chứa model |
| `onepeace_video_k400.pth` (6.6 GB) | upload lên Drive, thư mục tuỳ ý | checkpoint visual |
| `one-peace-audio.pt` (5.7 GB) | upload lên Drive, thư mục tuỳ ý | checkpoint audio (không cần `one-peace.pt` 15.5 GB) |
| `YouCookII/videos` (29 GB) | upload lên Drive | các file `<video_id>.mp4` **còn âm thanh** (hình cho visual, tiếng cho audio) |

Ví dụ bố cục (đặt khác cũng được, chỉ cần sửa đường dẫn ở bước 1.2):
```
MyDrive/
└── KL/
    ├── One_Peace/                       ← git clone từ GitHub
    │   ├── run_colab.ipynb
    │   ├── extract_video_features.py, extract_audio_features.py, ...
    │   └── annotations/youcookii_all.json
    ├── checkpoints/
    │   ├── onepeace_video_k400.pth
    │   └── one-peace-audio.pt
    └── YouCookII/
        └── videos/*.mp4
```
Clone code vào Drive (chạy trong một ô Colab sau khi mount Drive):
```bash
!git clone <URL repo GitHub của bạn> /content/drive/MyDrive/KL/One_Peace
```
Thư mục feature (`MyDrive/KL/feats/youcookii`) sẽ được tạo tự động.

Không cần tách audio riêng: phần audio tự lấy track âm thanh trong mp4, giải mã và resample giống hệt
`librosa.load(sr=16000)` của code gốc ONE-PEACE (đã kiểm tra: trùng tuyệt đối). Lưu ý: trình xem video của
VS Code không phát tiếng, nên đừng dựa vào đó để kết luận mp4 không có âm thanh.

### 0.2 Dung lượng Drive cần thêm cho feature
Với stride 0.5 s: khoảng **6 GB visual + 6 GB audio**.

---

## 1. Chạy bằng notebook (khuyên dùng)

### 1.1 Mở notebook
1. Trên Google Drive, chuột phải `One_Peace/run_colab.ipynb` → **Mở bằng → Google Colaboratory**.
2. **Runtime → Change runtime type → GPU**, nên chọn **A100** (T4 chạy được nhưng rất chậm).

### 1.2 Sửa đường dẫn — chỉ ở ô số 2
```python
ONE_PEACE_DIR = "/content/drive/MyDrive/KL/One_Peace"  # thư mục code (bản clone)
VIDEO_CKPT = "/content/drive/MyDrive/KL/checkpoints/onepeace_video_k400.pth"  # checkpoint visual
AUDIO_CKPT = "/content/drive/MyDrive/KL/checkpoints/one-peace-audio.pt"  # checkpoint audio
VIDEO_DIR = "/content/drive/MyDrive/KL/YouCookII/videos"  # thư mục chứa .mp4 (còn âm thanh)
FEAT_DIR = "/content/drive/MyDrive/KL/feats/youcookii"  # nơi ghi feature .npy
```
Notebook tự copy 2 checkpoint từ Drive ra `/content/ckpt/` (ô A1, B1) để nạp nhanh hơn;
hai biến `VIDEO_CKPT_LOCAL`, `AUDIO_CKPT_LOCAL` ở cuối ô 2 không cần sửa.

Cách lấy đúng đường dẫn: sau khi chạy ô 1, bấm biểu tượng 📁 bên trái Colab → mở `drive/MyDrive/...` →
chuột phải vào thư mục → **Copy path**.

Các tham số còn lại trong ô 2 **giữ nguyên** để sát bài báo:

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `STRIDE` | 8 | bước cửa sổ visual = 8 frame = 0.5 s (như ActivityNet trong bài báo) |
| `STRIDE_SEC` | `STRIDE / 16` | bước audio, tự khớp với visual — không sửa riêng |
| `BATCH_VIDEO` | 16 | A100: 16; T4: 8. Giảm nếu báo hết bộ nhớ GPU (`OutOfMemoryError`) |
| `DTYPE_VIDEO` | `fp16` | cos ≥ 0.9999 so với fp32; chỉ đổi sang `bf16` nếu báo NaN/inf |
| `COMPILE_VIDEO` | `True` | torch.compile, nhanh hơn; nếu lỗi biên dịch thì đặt `False` |
| `BATCH_AUDIO` | 64 | giảm nếu hết bộ nhớ GPU |
| `NUM_SHARDS`, `SHARD_ID` | 1, 0 | chia việc cho nhiều phiên Colab (mục 3) |

### 1.3 Chạy lần lượt các ô

**Ô 1 → 3 (luôn chạy đầu tiên):**
- Ô 1 hỏi quyền truy cập Drive → bấm cho phép.
- Ô 3 phải in `OK` cho mọi dòng (thư mục code, 2 checkpoint, thư mục video, annotation) và in số video khoảng 1500. Nếu có dòng `THIẾU` thì sửa lại ô 2.

**Phần A — Visual:**

| Ô | Việc | Kết quả mong đợi |
|---|---|---|
| A1 | Cài `einops`, copy checkpoint ra `/content` | vài phút (copy 6.6 GB) |
| A2 | Kiểm tra tiền xử lý | dòng `clip @92.0s ... making a sandwich (1.00)`, dòng `fp16 vs fp32: cos min ~0.9999x` |
| A3 | Đo tốc độ trên 400 clip | các dòng tiến độ in `x.xx clip/s` → ghi lại con số ở dòng cuối (các batch đầu có cả thời gian biên dịch) |
| A4 | **Chạy thật** | mỗi video in 1 dòng `[i/N] <id>: (T, 1536) ...` |

Ước lượng thời gian A4 ≈ `945000 / (clip/s ở A3)` giây cho toàn bộ 1500 video.
Nếu quá lâu so với thời lượng 1 phiên Colab, xem mục 3.

**Phần B — Audio** (có thể làm ở phiên khác, không phụ thuộc phần A):

| Ô | Việc | Kết quả mong đợi |
|---|---|---|
| B1 | Tạo môi trường Python 3.10, cài thư viện, copy checkpoint | ~5 phút; dòng cuối phải in `torch 2.1.2+cu121 cuda True` |
| B2 | **Chạy thật** | `Nạp model xong ... dtype=fp16`, rồi mỗi video 1 dòng `[i/N] <id>: (T, 1536) ...` |

**Phần C — Kiểm tra** (sau khi xong cả A và B):

Ô C in ra các dòng như sau (số liệu minh hoạ):
```
1500 video trong annotation, 1500 hợp lệ, thiếu file: 0, sai shape: 0
|T_visual - T_audio|: TB 0.xx, max 1-2 bước
chuẩn L2 visual TB ~20-30 ..., audio TB 1.000
```

---

## 2. Khi Colab bị ngắt giữa chừng

Không mất kết quả: mỗi video được ghi ngay lên Drive khi xong.
1. Kết nối lại runtime.
2. Chạy lại **ô 1 → 3**.
3. Chạy lại **ô cài đặt** của phần đang làm (A1 hoặc B1).
4. Chạy lại **ô chạy thật** (A4 hoặc B2). Video nào đã có file `.npy` sẽ tự bỏ qua.

---

## 3. Chạy song song nhiều phiên (để xong nhanh hơn)

Chia danh sách video thành `NUM_SHARDS` phần; mỗi phiên Colab (có thể dùng tài khoản khác, cùng thư mục Drive
được chia sẻ) xử lý một phần. Ví dụ 3 phiên:

| Phiên | Ô 2 |
|---|---|
| 1 | `NUM_SHARDS = 3`, `SHARD_ID = 0` |
| 2 | `NUM_SHARDS = 3`, `SHARD_ID = 1` |
| 3 | `NUM_SHARDS = 3`, `SHARD_ID = 2` |

Mọi phiên phải dùng **cùng `NUM_SHARDS`** và cùng `FEAT_DIR`. Có thể cho một phiên chạy visual, một phiên chạy audio cùng lúc.

---

## 4. Xử lý lỗi thường gặp

| Hiện tượng | Nguyên nhân / cách xử lý |
|---|---|
| Ô 3 báo `THIẾU` | Sai đường dẫn ở ô 2 → dùng **Copy path** như mục 1.2 |
| `torch.cuda.OutOfMemoryError` | Giảm `BATCH_VIDEO` (hoặc `BATCH_AUDIO`) ở ô 2, chạy lại ô 2 rồi ô chạy thật |
| Ô A2 không ra "making a sandwich" | Tiền xử lý hoặc checkpoint sai. Kiểm tra `onepeace_video_k400.pth` đủ 6 629 055 726 byte (ô A1 in kích thước) |
| Ô A1/B1 copy checkpoint rất lâu hoặc lỗi | Drive đang đồng bộ file chưa xong; đợi upload xong hẳn. File copy dở thì xoá `/content/ckpt/*` rồi chạy lại ô |
| B1 báo lỗi cài `omegaconf` | Chưa hạ pip → chạy lại cả ô B1 từ đầu |
| B2 báo `No module named ...` | Chạy nhầm bằng `python` thay vì `/content/op310/bin/python`, hoặc chưa chạy B1 trong phiên này |
| B2 báo `No module named 'pkg_resources'` | `setuptools` quá mới → chạy `!/content/op310/bin/python -m pip install "setuptools<81"` |
| B2 báo `No module named 'resampy'` | thiếu gói cho `--res_type kaiser_best` → `!/content/op310/bin/python -m pip install resampy` (hoặc dùng `--res_type soxr_hq`) |
| B2 chạy bằng CPU dù đã chọn GPU | `timm` đã kéo về bản torch CPU → chạy lại ô B1 từ đầu (ô này cài torch và torchvision cùng lệnh) |
| B2 báo `checkpoint không khớp model` | `AUDIO_CKPT` trỏ sai file → phải là `one-peace-audio.pt` tạo bởi `slim_audio_checkpoint.py` |
| Có file `failed_video_shard*.txt` / `failed_audio_shard*.txt` trong `FEAT_DIR` | Danh sách video lỗi (thường do mp4 hỏng). Mở file xem lý do; chạy lại ô chạy thật để thử lại các video đó |
| Có file `no_audio_track_shard*.txt` | Video không có âm thanh → đã được thay bằng im lặng, không cần làm gì |
| Ô C báo thiếu file | Một phần (A hoặc B) chưa xong, hoặc có shard chưa chạy |

---

## 5. Chạy không dùng notebook

Mọi đường dẫn là tham số dòng lệnh, không có file cấu hình. Xem đủ tham số bằng `python <script>.py --help`.

```bash
cd <thư mục One_Peace>

# Visual (Python bất kỳ có torch + einops + opencv; cần ffmpeg)
python extract_video_features.py \
    --checkpoint <đường dẫn onepeace_video_k400.pth> \
    --video_dir  <thư mục mp4> \
    --output_dir <thư mục feature> \
    --ids_from   annotations/youcookii_all.json \
    --stride 8 --batch_size 16 --compile

# Audio (Python 3.10 + thư viện như ô B1; cần repo ONE-PEACE)
python extract_audio_features.py \
    --onepeace_repo <thư mục ONE-PEACE đã git clone> \
    --checkpoint <đường dẫn one-peace-audio.pt> \
    --video_dir  <thư mục mp4> \
    --output_dir <thư mục feature> \
    --ids_from   annotations/youcookii_all.json \
    --stride_sec 0.5 --batch_size 64

# Kiểm tra
python check_features.py --anno annotations/youcookii_all.json --feat_dir <thư mục feature> --stride 8
```
Chạy trên CPU vẫn được (tự nhận khi không có GPU) nhưng rất chậm: ~70 s cho mỗi clip visual, ~1.2 lần thời lượng video cho audio.

Hai tham số audio đáng lưu ý:
- `--res_type` (mặc định `kaiser_best`, cần gói `resampy`): bộ lọc resample. Đây là yếu tố ảnh hưởng lớn nhất
  khi đối chiếu với feature của tác giả (cos 0.93 so với 0.84 của `soxr_hq`) — xem mục 9 của [README.md](README.md).
- `--file_prefix`: dùng khi tên file có tiền tố mà id trong annotation không có, ví dụ tập validation của DESED
  (`--file_prefix Y` cho file `Y<id>.wav`).

Không chạy 2 tiến trình trích xuất trên cùng một GPU 12 GB (tràn VRAM lúc nạp checkpoint). Chia việc bằng
`--num_shards` thì mỗi shard nên ở một phiên Colab / một GPU riêng.

---

## 6. Sau khi có feature → đưa vào UniAV

1. Copy (hoặc tạo symlink) feature vào `UniAV-fixed/data/youcookii/av_features/`.
2. Copy `annotations/youcookii_all.json` vào `UniAV-fixed/data/youcookii/annotations/`.
3. Thêm khối cấu hình YouCookII vào file config của UniAV (mẫu ở mục 6 của [README.md](README.md)).
