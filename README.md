# Leng Keng - Nhận diện cử chỉ tay (Hand Gesture Recognition)

Dự án nhận diện cử chỉ tay theo thời gian thực bằng webcam, dùng **MediaPipe** để lấy điểm mốc bàn tay và **scikit-learn** để phân loại cử chỉ. Có thể dùng làm đầu vào điều khiển game (mỗi cử chỉ = một "chiêu").

## Luồng hoạt động

Pipeline gồm 3 bước theo thứ tự:

```
collect_data.py  ->  train_model.py  ->  main.py
   (thu data)         (huấn luyện)       (chạy realtime)
```

1. **Thu thập dữ liệu** cử chỉ vào `gestures.csv`
2. **Huấn luyện** mô hình phân loại, lưu ra `gesture_model.joblib`
3. **Chạy nhận diện** realtime bằng model đã huấn luyện

## Cài đặt

Yêu cầu Python 3.10+. Nên tạo môi trường ảo:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

Cần có webcam để chạy.

## Cách sử dụng

### Bước 1 — Thu thập dữ liệu

```bash
python collect_data.py
```

- Giơ cử chỉ trước webcam.
- **Giữ phím số (0–9)** để ghi mẫu cho chiêu tương ứng (mỗi frame = 1 mẫu).
- Nên thu mỗi chiêu khoảng **200–300 mẫu**, đổi góc / khoảng cách / tay trái–phải cho đa dạng.
- Nhấn `q` (hoặc `Esc`) để thoát. Dữ liệu được ghi **nối tiếp** vào `gestures.csv` (chạy lại không mất data cũ).

Danh sách chiêu và phím tương ứng được định nghĩa trong `gesture_utils.py` (biến `GESTURES`). Muốn thêm/đổi chiêu thì sửa danh sách này (tối đa 10 chiêu, ứng với phím 0–9).

### Bước 2 — Huấn luyện mô hình

```bash
python train_model.py
```

Đọc `gestures.csv`, huấn luyện `MLPClassifier`, in báo cáo độ chính xác trên tập test và lưu model ra `gesture_model.joblib`.

### Bước 3 — Nhận diện realtime

```bash
python main.py
```

Bật webcam, hiển thị tên chiêu nhận diện được kèm độ tin cậy. Nhấn `q` (hoặc `Esc`) để thoát.

Đặt logic game của bạn tại chỗ đánh dấu `>>> DAT LOGIC GAME TAI DAY` trong `main.py`, ví dụ khi `vote == 0` thì gọi hàm tung chiêu 1.

## Cấu trúc thư mục

| File | Vai trò |
|------|---------|
| `gesture_utils.py` | Dùng chung: danh sách `GESTURES` và hàm `normalize_landmarks` chuẩn hóa 21 điểm mốc tay thành vector 42 chiều (bất biến với vị trí và kích thước tay). |
| `collect_data.py` | Thu thập dữ liệu cử chỉ từ webcam, ghi vào `gestures.csv`. |
| `train_model.py` | Huấn luyện mô hình phân loại từ `gestures.csv`, xuất ra `gesture_model.joblib`. |
| `main.py` | Nhận diện cử chỉ realtime từ webcam. Có làm mượt kết quả qua nhiều frame, ngưỡng tin cậy, và thử cả bản lật gương của bàn tay. |
| `hand_landmarker.task` | Model MediaPipe (có sẵn) để phát hiện điểm mốc bàn tay. **Bắt buộc** cho cả bước thu data và bước chạy. |
| `gestures.csv` | Dữ liệu cử chỉ đã thu (do `collect_data.py` sinh ra). |
| `gesture_model.joblib` | Mô hình phân loại đã huấn luyện (do `train_model.py` sinh ra). |
| `requirements.txt` | Danh sách thư viện phụ thuộc. |

## Tham số có thể tinh chỉnh

Trong `main.py`:

- `CONF_THRESHOLD` (mặc định `0.8`) — dưới ngưỡng này coi như không rõ chiêu.
- `SMOOTH_FRAMES` (mặc định `7`) — số frame gần nhất dùng để bình chọn kết quả (làm mượt).

Trong `collect_data.py` và `main.py`:

- `CAM_W`, `CAM_H` — độ phân giải webcam.

## Ghi chú

- `gestures.csv` và `gesture_model.joblib` là file **sinh ra**. Nếu muốn repo gọn, có thể không commit `gesture_model.joblib` (người khác clone về tự chạy `train_model.py` để tạo lại).
- Thư mục `.vscode/` chứa cấu hình đường dẫn `.venv` theo máy cá nhân; cân nhắc thêm vào `.gitignore` nếu chia sẻ cho người khác.
