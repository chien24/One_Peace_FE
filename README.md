# Trích xuất đặc trưng ONE-PEACE cho UniAV (YouCookII)

## 1. Tác giả UniAV xử lý dữ liệu như thế nào?

Repo `UniAV-fixed` **không có code trích xuất đặc trưng**: model chỉ đọc file `.npy` có sẵn
(`libs/datasets/anet.py`, dòng 190–203). Cách làm chỉ được mô tả trong README và bài báo:

| | Visual | Audio |
|---|---|---|
| Encoder | ONE-PEACE vision encoder **fine-tune trên Kinetics-400** (`onepeace_video_k400.pth`) | ONE-PEACE audio encoder (`one-peace.pt`) |
| Lấy mẫu | 16 fps, cạnh ngắn resize về 256, center crop 256×256 | 16 kHz |
| Cửa sổ | 16 frame liên tiếp (= 1 s) | 1 s |
| Bước | 8 frame = 0.5 s (ActivityNet) / 4 frame = 0.25 s (DESED, UnAV-100) | 0.5 s / 0.25 s |
| Đầu ra | `<id>_one_peace_video_finetune.npy`, (T, 1536) | `<id>_one_peace_audio.npy`, (T, 1536) |

Sau đó UniAV: ActivityNet → nội suy về độ dài cố định 256; UnAV-100 / DESED → pad/crop về 256 / 64.
Nếu visual và audio lệch số bước, loader cắt cả hai về độ dài ngắn hơn.

Những gì mình đã **kiểm tra trực tiếp trên feature DESED tác giả công bố**:
- clip 10 s → 37 bước cho cả hai modality = `(160 − 16) / 4 + 1`. Công thức cửa sổ trượt ở trên là đúng;
- audio: float32, **chuẩn L2 = 1** (đầu ra `extract_audio_features` của ONE-PEACE đã normalize);
- visual: float64, **không normalize**, chuẩn L2 ≈ 20–32. Đây là trung bình CLS token của 16 frame
  từ backbone, đúng là đầu vào `I3DHead` khi fine-tune K400. Script ở đây lưu float32, vì loader của
  UniAV ép kiểu về float32 ngay khi đọc (`np.load(...).astype(np.float32)`), nên kết quả như nhau.

## 2. Các lựa chọn để bám sát bản gốc (mặc định của mọi script)

Bài báo không ghi chi tiết cài đặt, nên mỗi bước được làm **giống code gốc mà tác giả phải dùng**
(pipeline test K400 của ONE-PEACE cho visual, `OnePeaceHubInterface` cho audio).
Ảnh hưởng đo trên `GLd3aX16zBg.mp4` (CPU):

| Bước | Mặc định (sát gốc) | Phương án nhanh | Ảnh hưởng đo được |
|---|---|---|---|
| Resize/crop frame | ffmpeg chỉ giải mã + đổi 16 fps, rồi **cv2 bilinear + công thức làm tròn của mmcv + CenterCrop của mmaction2** | `--resize_backend ffmpeg` | ffmpeg làm tròn chiều rộng lên số chẵn (456 thay vì 455) → crop lệch 1 pixel; feature: cos 0.993, sai khác tương đối **11.8%** |
| Đọc audio | Giải mã PCM 16-bit ở sample rate gốc, mono = trung bình kênh, **librosa soxr_hq** → **trùng tuyệt đối** với `librosa.load(wav, sr=16000)` | `--resampler ffmpeg` | feature: cos TB 0.976, **thấp nhất 0.84** |
| Độ chính xác video | GPU: **fp16** (`--dtype auto`); CPU: fp32 | `--dtype fp32` (sát nhất, chậm ~3.4 lần), `bf16` | Đo trên RTX 3060, so với fp32 (tắt TF32): fp16 cos min **0.99995**, sai khác tương đối 7.3e-3; bf16 cos min 0.998 (5.1e-2) |
| Độ chính xác audio | GPU: **fp16** (`--dtype auto`); CPU: float32 | `--dtype float32` (sát nhất, chậm ~2.6 lần) | Đo trên RTX 3060: fp16 cos min **0.999996** so với float32 |
| Attention video | `scaled_dot_product_attention` (kernel memory-efficient) | `--no_sdpa` (matmul gốc) | fp32: sai khác 9e-7 so với code cũ → tương đương |
| Checkpoint audio | `one-peace-audio.pt` giữ **fp32** (5.72 GB) | `slim_audio_checkpoint.py --fp16` (2.86 GB) | cos 1.00000 |

Những điểm **không thể biết chắc** vì tác giả không công bố code: cách đưa video về 16 fps
(ở đây dùng bộ lọc `fps=16` của ffmpeg) và việc họ đọc audio bằng `librosa.load` trên mp4 hay trên
wav tách trước. Hai cách này cho cùng một kết quả vì đều đi qua PCM 16-bit.

## 3. Nội dung thư mục

| File | Việc |
|---|---|
| `onepeace_video_backbone.py` | Backbone video ONE-PEACE tách khỏi mmaction/mmcv, nạp `onepeace_video_k400.pth` |
| `video_io.py` | Đọc frame (resize kiểu mmaction2) / audio (kiểu librosa) dạng stream, cửa sổ trượt, chia shard |
| `extract_video_features.py` | Trích xuất visual → `*_one_peace_video_finetune.npy` |
| `extract_audio_features.py` | Trích xuất audio → `*_one_peace_audio.npy` (dùng code gốc ONE-PEACE), audio lấy thẳng từ mp4 |
| `slim_audio_checkpoint.py` | Tách nhánh audio từ `one-peace.pt` |
| `onepeace_video_k400.pth` | Checkpoint video K400 (1.66B tham số) |
| `one-peace.pt` | Checkpoint pretrain ONE-PEACE (3.89B tham số) — chỉ cần để tạo file dưới |
| `one-peace-audio.pt` | **Đã tạo sẵn**: nhánh audio, fp32, 1.53B tham số — dùng cho trích xuất audio |
| `build_youcookii_annotations.py` | `youcookii_{train,val}_preprocess.json` → định dạng annotation của UniAV |
| `check_features.py` | Kiểm tra shape, độ lệch visual/audio, chuẩn L2 |
| `sanity_check_video.py` | Chạy vài clip qua head K400 để xác nhận tiền xử lý; so SDPA/bmm, fp16/fp32 |
| `annotations/youcookii_all.json` | Annotation đã tạo (1106 video train / 394 video val) |
| `label_map_k400.txt` | Tên 400 lớp Kinetics (chỉ để kiểm tra) |

### Đã chạy thử (CPU, máy local, trên `GLd3aX16zBg.mp4`)
- **Visual:** clip giây 92 (*"spread margarine on two slices of white bread"*) và giây 115 → head K400
  **"making a sandwich" (1.00)**; clip giây 30 → "making a sandwich" (0.84). Chuẩn feature 28–30,
  khớp khoảng của DESED. 241.65 s → 3866 frame = 241.65 × 16.
- **Audio** (checkpoint thật, fp32, librosa): 241.65 s → (482, 1536), khớp 482 bước visual (stride 0.5 s),
  chuẩn L2 = 1, 293 s trên CPU. So với 1152 cửa sổ audio DESED của tác giả: cos giữa hai vector trung bình
  = 0.57, cos từng cặp TB = 0.17 (trong nội bộ DESED là 0.20). Với vector ngẫu nhiên cả hai ≈ 0, nên feature
  nằm cùng không gian embedding với feature tác giả. Đây là bằng chứng gián tiếp, vì không có audio DESED gốc.
- Loader audio (dựng model trên `meta` rồi gán trọng số) cho kết quả trùng tuyệt đối với loader fairseq gốc
  (thử trên checkpoint nhỏ), nhưng không cấp phát thêm ~6 GB tham số ngẫu nhiên.

## 4. Bước thời gian cho YouCookII

YouCookII: video dài TB ~315 s, tổng ~135 giờ cho 1500 video. Bài toán giống ActivityNet
(localization trên video dài), nên cấu hình sát bài báo nhất là **stride 8 frame / 0.5 s + `force_upsampling` về 256**.

| Bước | Số clip (1500 video) | Dung lượng feature (mỗi modality, float32) |
|---|---|---|
| 0.25 s (stride 4, như DESED/UnAV-100) | ~1.89 triệu | 11.6 GB |
| **0.5 s (stride 8, như ActivityNet)** | **~945 nghìn** | **5.8 GB** |

⚠️ **Giới hạn thật là thời gian GPU, không phải dung lượng.** Mỗi clip visual là 16 × 257 token qua model
1.66B tham số, khoảng **17 TFLOP / clip**: tốc độ bị giới hạn bởi sức tính GPU chứ không phải dung lượng checkpoint
(CPU local: ~70 s/clip). Nên dùng **A100** và đo trước bằng `--limit 1 --max_clips 400` (script in `clip/s`).
Chi tiết tốc độ đo được ở mục 7. Dùng `--num_shards/--shard_id` để chia nhiều phiên;
script bỏ qua video đã có file nên phiên bị ngắt chỉ cần chạy lại.
Audio nhẹ hơn nhiều (mỗi cửa sổ chỉ ~50 token).

## 5. Chạy trên Google Colab

Upload lên Google Drive: `YouCookII/videos` (29 GB) và thư mục `One_Peace` này. Cần
`onepeace_video_k400.pth` (6.6 GB) và `one-peace-audio.pt` (5.7 GB); **không cần** `one-peace.pt`.

### 5.1 Visual (môi trường Python mặc định của Colab)

```bash
!pip install -q einops
!cp /content/drive/MyDrive/One_Peace/onepeace_video_k400.pth /content/   # nạp nhanh hơn đọc từ Drive
%cd /content/drive/MyDrive/One_Peace
# kiểm tra tiền xử lý + đo sai khác nếu muốn dùng fp16
!python sanity_check_video.py --checkpoint /content/onepeace_video_k400.pth \
    --video /content/drive/MyDrive/YouCookII/videos/GLd3aX16zBg.mp4 --start_sec 92 --compare_fp16
# đo tốc độ
!python extract_video_features.py --checkpoint /content/onepeace_video_k400.pth \
    --video_dir /content/drive/MyDrive/YouCookII/videos --output_dir /content/drive/MyDrive/feats/youcookii \
    --ids_from annotations/youcookii_all.json --stride 8 --batch_size 16 --compile --limit 1 --max_clips 400 --overwrite
# chạy thật (ví dụ chia 4 phiên: shard_id 0..3)
!python extract_video_features.py --checkpoint /content/onepeace_video_k400.pth \
    --video_dir /content/drive/MyDrive/YouCookII/videos --output_dir /content/drive/MyDrive/feats/youcookii \
    --ids_from annotations/youcookii_all.json --stride 8 --batch_size 16 --compile --num_shards 4 --shard_id 0
```
Lệnh đo tốc độ chỉ lấy 400 clip đầu, nên phải có `--overwrite` ở lệnh chạy thật hoặc xoá file `.npy` đó.
Nếu hết VRAM thì giảm `--batch_size`.

### 5.2 Audio (cần môi trường Python 3.10 riêng)

Code ONE-PEACE dùng fairseq cũ (`hydra-core 1.0.7`, `omegaconf 2.0.6`): không chạy trên Python 3.12
của Colab, và **pip ≥ 24.1 từ chối cài `omegaconf 2.0.6`** (metadata cũ) nên phải hạ pip về 24.0.
Không cần `pip install ./fairseq`: script tự thêm `ONE-PEACE/fairseq` vào `sys.path`.

```bash
%cd /content
!git clone --depth 1 https://github.com/OFA-Sys/ONE-PEACE
!pip install -q uv && uv venv --seed --python 3.10 /content/op310
!/content/op310/bin/python -m pip install -q "pip==24.0"
!/content/op310/bin/python -m pip install -q torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
!/content/op310/bin/python -m pip install -q "setuptools<81" "numpy<2" hydra-core==1.0.7 omegaconf==2.0.6 \
    antlr4-python3-runtime==4.8 bitarray sacrebleu tabulate regex timm==0.6.11 iopath tensorboardX pydub \
    librosa==0.10.0 resampy soundfile soxr einops opencv-python-headless scipy tqdm pillow imageio-ffmpeg
!cp /content/drive/MyDrive/One_Peace/one-peace-audio.pt /content/

%cd /content/drive/MyDrive/One_Peace
!/content/op310/bin/python extract_audio_features.py --onepeace_repo /content/ONE-PEACE \
    --checkpoint /content/one-peace-audio.pt \
    --video_dir /content/drive/MyDrive/YouCookII/videos --output_dir /content/drive/MyDrive/feats/youcookii \
    --ids_from annotations/youcookii_all.json --stride_sec 0.5 --batch_size 64
```
`librosa==0.10.0` là phiên bản ONE-PEACE ghi trong `requirements.txt`. Hai ràng buộc dễ bỏ sót:
`setuptools<81` (bản mới bỏ `pkg_resources`, mà librosa 0.10.0 vẫn import) và `torch`/`torchvision` phải
cài chung một lệnh (nếu không `timm` kéo về bản torch mới nhất, là bản CPU).
`resampy` cần cho `--res_type kaiser_best` (mặc định) — xem mục 9.
`--stride_sec` phải bằng `stride / 16` của visual. Video không có track âm thanh được thay bằng im lặng
cùng độ dài và ghi vào `no_audio_track_shard*.txt`.

Tạo lại `one-peace-audio.pt` (mọi môi trường có torch, không cần fairseq):
`python slim_audio_checkpoint.py --input one-peace.pt --output one-peace-audio.pt`.
Script đọc thẳng từng tensor trong file zip của checkpoint; không dùng `torch.load(mmap=True)` vì trên
Windows cách đó lỗi *access violation* với file 15.5 GB.

Thông tin checkpoint `one-peace.pt` (đã kiểm tra): chỉ có `cfg` + `model`; `task = audio_text_pretrain`,
`head_type = val`, `bpe_dir = one_peace/utils/BPE` (tương đối), `model._name = one_piece_pretrain`
(tên gõ sai, fairseq không đăng ký), thiếu `logit_scale` (`reset_logit_scale = True`). Vì vậy
`extract_audio_features.py` override `model._name = one_peace_retrieval`, `head_type = audio`, `bpe_dir`
tuyệt đối và khởi tạo lại `logit_scale` (không dùng khi trích feature) — giống `from_pretrained` gốc.

Lưu ý: thư mục `YouCookII/audio` hiện có là **wav cắt theo từng đoạn** (`<id>_<k>.wav`), không dùng được
cho UniAV vì UniAV cần audio của **toàn bộ video**, nên script lấy audio trực tiếp từ mp4.

### 5.3 Kiểm tra

```bash
!python check_features.py --anno annotations/youcookii_all.json --feat_dir /content/drive/MyDrive/feats/youcookii --stride 8
```
Kết quả mong đợi: không thiếu file, `|T_visual − T_audio|` ≈ 0–2, chuẩn visual ~20–30, audio ~1.0.

## 6. Annotation và cấu hình UniAV

```bash
python build_youcookii_annotations.py \
    --train_json D:/Học/KL/Data/YouCookII/metadata/youcookii_train_preprocess.json \
    --val_json   D:/Học/KL/Data/YouCookII/metadata/youcookii_val_preprocess.json \
    --output annotations/youcookii_all.json --video_dir D:/Học/KL/Data/YouCookII/videos
```
- YouCookII **không có nhãn lớp**, chỉ có câu mô tả từng bước. Mặc định `--label_mode step` gán 1 lớp
  `"cooking step"` (num_classes = 1). `--label_mode recipe` dùng loại món (89 lớp). Câu mô tả gốc được giữ
  trong trường `sentence`.
- 33 video train và 10 video val trong metadata không có file mp4 nên bị bỏ (330 đoạn).
- Có thể thêm `--feat_dir <thư mục feature>` để chỉ giữ các video đã trích xuất xong.

Gợi ý khối cấu hình (dựa theo TASK1 ActivityNet trong `configs/multi_task_anet_unav_dcase.yaml`,
**chưa chạy thử với UniAV**). Đặt feature vào `./data/youcookii/av_features`:
```yaml
TASK1:
  task_type: TAL
  dataset_name: anet
  train_split: ['training']
  val_split: ['validation']
  test_split: ['validation']
  dataset: {
    json_file: ./data/youcookii/annotations/youcookii_all.json,
    feat_folder: ./data/youcookii/av_features,
    feat_stride: 8,
    num_frames: 16,
    default_fps: 16,
    trunc_thresh: 0.5,
    num_classes: 1,
    max_seq_len: 256,
    max_buffer_len_factor: 1.0,
    force_upsampling: True,
    class_aware: False,
  }
```
Không đặt `file_prefix` (ActivityNet dùng `v_`, YouCookII thì không). Loader `anet` đánh giá mAP ở
tIoU 0.5:0.95; các bài trên YouCookII thường báo thêm tIoU 0.3/0.5/0.7.

## 7. Tốc độ và các tối ưu đã làm (visual)

Đo trên RTX 3060 12 GB, PyTorch 2.11, video 640x360, stride 8, chỉ tính forward model:

| Cấu hình | clip/s | VRAM đỉnh | cos min so với fp32 |
|---|---|---|---|
| Code cũ, fp32 (tắt TF32) | 0.36 | 9.6 GB | 1 (chuẩn) |
| Code cũ, fp16 | 0.89 | 5.0 GB | 0.99995 |
| **Code mới, fp16** | **1.22** | **4.0 GB** | **0.99995** |
| Code mới, fp16 + `--compile` | 1.32 | — | 0.99995 |

Code cũ và code mới cho feature trùng nhau (fp32: sai khác 9e-7; chạy end-to-end fp16 trên 200 clip: cos min 0.999998).

Các thay đổi:
1. **Attention không gian rơi về nhánh "math" fp32.** Bias vị trí tương đối được tạo bằng `permute` nên không liền bộ nhớ,
   cộng với tensor 3D, khiến `scaled_dot_product_attention` không dùng được kernel memory-efficient: cả 40 layer tính
   attention 257x257 bằng GEMM fp32 + softmax riêng (~25% thời gian mỗi forward khi chạy fp16). Nay q/k/v là tensor 4D,
   bias có dạng (1, H, L, L) liền bộ nhớ và được căn lề. Đây là nguyên nhân chính của mức tăng +38%.
2. Bias không còn bị nhân bản thành (B·T·H, 257, 257) và ép kiểu lại ở mỗi layer (~400 MB mỗi lần với batch 4).
3. Tensor batch-first, chỉ LayerNorm token CLS ở cuối, giữ feature trên GPU tới khi xong video
   (không đồng bộ CPU–GPU ở mỗi batch), dùng pinned memory, mỗi video chỉ `probe` một lần.
4. `prefetch` gom batch ngay ở thread nền; sửa lỗi thread nền bị kẹt (và ffmpeg chạy ngầm) khi dừng sớm với `--max_clips`.
5. `--compile` (torch.compile): gộp LayerNorm/GELU/phép cộng; batch cuối mỗi video được pad cho đủ để không phải biên dịch lại.

Sau khi sửa, ~80% thời gian GPU nằm ở phép nhân ma trận và các GEMM đã chạy gần trần fp16 của RTX 3060 (~25 TFLOPS).
Muốn nhanh hơn nữa chỉ còn cách dùng GPU mạnh hơn (A100) hoặc chia nhiều phiên (`--num_shards`).

## 8. Kiểm chứng và tốc độ phần audio

Đo trên RTX 3060, file wav 10 s (DESED), cửa sổ 1 s / bước 0.5 s:

| Kiểm tra | Kết quả |
|---|---|
| So với `OnePeaceHubInterface.process_audio` gốc (cùng waveform 16000 mẫu) | **cos = 1.000000** |
| `load_audio` (ffmpeg + librosa resample) so với `librosa.load(sr=16000)` | trùng từng mẫu (max \|diff\| = 0) |
| Chuẩn L2 của feature | 1.0000 (khớp feature DESED của tác giả) |
| Token mỗi cửa sổ 1 s | 49 |
| fp16 so với float32 | cos min **0.999996** |
| Tốc độ float32 / fp16 (batch 64) | 40 / 104 cửa sổ/s (VRAM 6.1 GB / ~3 GB) |
| Thay attention bmm bằng SDPA | không nhanh hơn (117 so với 117 cửa sổ/s) vì chuỗi chỉ 50 token, nên giữ nguyên code gốc |

Thay đổi: mặc định `--dtype auto` (fp16 trên GPU), và audio của video kế tiếp được giải mã ở thread nền.
Với video 98 s, giải mã mất 1.5 s còn GPU mất 1.9 s, nên chạy song song rút thời gian mỗi video
từ ~3.4 s xuống 1.8 s.

## 9. Đối chiếu với feature DESED của tác giả UniAV

Dữ liệu: 40 clip 10 s của tập `eval/public` (DESED), file wav trùng MD5 với bản phân phối chính thức
(ngày 05/2019) nên **đầu vào giống hệt của tác giả**. Cấu hình: cửa sổ 1 s, bước 0,25 s, fp16.

| Bộ lọc resample (`--res_type`) | cos TB | trung vị | min | số clip > 0.9 |
|---|---|---|---|---|
| **kaiser_best** (mặc định của librosa < 0.10 — **mặc định của script**) | **0.927** | 0.936 | 0.792 | 34/40 |
| soxr_hq (mặc định của librosa >= 0.10) | 0.840 | 0.843 | 0.696 | 7/40 |

Bộ lọc resample là yếu tố lớn nhất. Quét toàn bộ bộ lọc librosa (soxr_qq/lq/mq/hq/vhq, kaiser_best/fast,
polyphase, fft, sinc_window) thì `kaiser_best` cho kết quả tốt nhất, `soxr_qq` tệ nhất (0.58–0.78).
`torchaudio.functional.resample` (0.59–0.94) và `ffmpeg -ar 16000` đều kém hơn librosa.

Các giả thuyết khác **đã loại trừ** (mọi phương án đều thấp hơn cấu hình hiện tại):

| Thử nghiệm | cos TB |
|---|---|
| layer_norm cả clip / không chuẩn hoá | 0.60 / 0.34 |
| Cửa sổ 0,5 s / 2 s | 0.66 / 0.56 |
| Một lần forward cả clip rồi gộp token | 0.14 |
| Pad cửa sổ lên 2–10 s kèm mask | 0.77–0.81 |
| Bỏ `audio_proj` / trung bình token thay CLS | ~0.00 / 0.35 |
| Chỉ lấy 1 kênh, lượng tử hoá int16, lệch pha ±10–50 ms | không cải thiện |
| Checkpoint `finetune_al_retrieval.pt` | 0.59–0.65 |
| Checkpoint `finetune_fsd50k.pt` (attn-pooling hoặc CLS) | ~0.00–0.03 |
| `resampy 0.2.2` thay cho 0.4.3 (cùng kaiser_best) | 0.77–0.91 (thấp hơn) |

Đã kiểm chứng:
- Pipeline trùng tuyệt đối với `process_audio` gốc khi cùng waveform (cos = 1.000000);
- `one-peace-audio.pt` **trùng từng bit** với nhánh audio của `one-peace.pt` chính thức (885/885 tensor,
  không thiếu/thừa khoá) — tức checkpoint không phải nguyên nhân;
- `slim_audio_checkpoint.py` đọc đúng: kiểm tra tương tự trên `finetune_al_retrieval.pt`;
- Nén lại AAC/Opus/MP3 hay đổi âm lượng chỉ làm lệch ~1% (cos 0.99).

**Phần còn lại (~0.07) chưa giải thích được.** Đầu vào, checkpoint và pipeline đều đã xác nhận trùng với
bản chính thức, nên khả năng còn lại là phiên bản librosa/resampy tác giả dùng hoặc một chi tiết tiền xử lý
họ không công bố. Với UniAV thì mức này không quan trọng, miễn là **mỗi dataset dùng một nguồn feature
nhất quán** (tự trích xuất cả visual lẫn audio cho YouCookII, đừng trộn với feature của tác giả).

`--res_type kaiser_best` cần gói `resampy`. Muốn bám sát ONE-PEACE (librosa 0.10) thì dùng `--res_type soxr_hq`.

## 10. Các trường hợp đã kiểm tra bằng thực nghiệm

| Trường hợp | Kết quả |
|---|---|
| Video không có track âm thanh | thay bằng im lặng cùng độ dài, ghi `no_audio_track_shard0.txt`, số bước khớp visual |
| Audio < 1 s (wav 0,4 s) / video < 16 frame (0,4 s) | lặp lại cho đủ 1 cửa sổ, ra `(1, 1536)` |
| `check_features.py` | `|T_visual − T_audio| = 0`, lệch thời lượng 0 s, báo đúng số file thiếu |
| `build_youcookii_annotations.py` | bỏ video không có mp4, bỏ đoạn độ dài ≤ 0, cắt đoạn vượt `duration`, `--label_mode recipe` + `--recipe_names` gán đúng tên món, `--feat_dir` lọc đúng |
| Chia shard (`--num_shards 2`) | 6 + 6 video, không trùng lặp, ghi cùng một thư mục |
| Bị ngắt giữa lúc chạy (kill tiến trình) | không để lại `.npy` hỏng (ghi tạm rồi rename), không còn ffmpeg chạy ngầm |
| Chạy lại sau khi ngắt | tự bỏ qua video đã có `.npy` |
| `--resize_backend ffmpeg` (visual) | cos 0.992 so với cv2 |
| `--resampler ffmpeg` (audio) | cos 0.935 so với librosa → **nên giữ librosa** |
| `--dtype bf16` (audio) | cos 0.999 so với fp16 |
| `--no_sdpa` (visual) | cos 0.999 so với SDPA |
| `slim_audio_checkpoint.py --fp16` | ra file 2.86 GB, feature **cos 1.000000** so với checkpoint fp32 |
| Kích thước `onepeace_video_k400.pth` | đúng 6 629 055 726 byte như tài liệu |

**Lưu ý VRAM:** không chạy được 2 tiến trình trích xuất audio trên cùng một GPU 12 GB (mỗi tiến trình cần
~6 GB lúc nạp checkpoint fp32 rồi mới chuyển fp16). Muốn chạy song song thì mỗi shard một GPU / một phiên Colab.
