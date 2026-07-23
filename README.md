# Leng Keng — Hand Magic (Nhận diện cử chỉ + Hiệu ứng phép thuật 3D)

Giơ cử chỉ tay trước webcam để tung "chiêu" với hiệu ứng 3D thời gian thực.
Dùng **MediaPipe** lấy 21 điểm mốc bàn tay, **scikit-learn** phân loại cử chỉ,
và **Ursina** (engine 3D trên Panda3D) dựng hiệu ứng (cầu lửa, khói, glow…).

## Luồng tổng quan

```
Webcam ─► MediaPipe (21 landmarks) ─► ML model (nhận diện chiêu) ─► Hiệu ứng 3D
```

Mỗi frame: đọc webcam → lấy điểm mốc tay → model đoán ra số chiêu (0–5) →
`main.py` gọi hiệu ứng tương ứng.

## Cài đặt

Yêu cầu Python 3.10+ và webcam. Nên dùng môi trường ảo:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

## Chạy

```bash
python main.py          # app chính (nhận diện + hiệu ứng)
```

Điều khiển: xòe bàn tay (Chiêu 1) để bắn cầu lửa liên tục · phím `b` bật/tắt
glow (bloom) · `Esc` để thoát.

Các file phụ trợ:

```bash
python gesture_demo.py  # chỉ nhận diện + in tên chiêu (không hiệu ứng), để test model
python collect_data.py  # thu dữ liệu cử chỉ mới
python train_model.py   # train lại model
```

## Cấu trúc file

| File | Vai trò |
|------|---------|
| `main.py` | **File chính.** Webcam, MediaPipe, ML nhận diện chiêu, scene Ursina, vòng lặp game. Đây là nơi *gắn* cử chỉ với hiệu ứng. |
| `Fireball3D.py` | **Module hiệu ứng.** Shader (bloom, lửa), texture, các particle (`Ember`, `Smoke`, `TrailPuff`), class `Fireball`. Không biết gì về webcam/ML. |
| `gesture_utils.py` | Dùng chung: danh sách `GESTURES` và `normalize_landmarks` (chuẩn hóa 21 điểm → vector 42 chiều). |
| `collect_data.py` | Thu dữ liệu cử chỉ, ghi nối tiếp vào `gestures.csv`. |
| `train_model.py` | Train `MLPClassifier` từ `gestures.csv` → `gesture_model.joblib`. |
| `gesture_demo.py` | Demo nhận diện thuần (không hiệu ứng), tiện để test độ chính xác model. |
| `hand_landmarker.task` | Model MediaPipe phát hiện điểm mốc tay (bắt buộc). |
| `gestures.csv` / `gesture_model.joblib` | Dữ liệu và model đã train (sinh ra). |

---

# 👉 HƯỚNG DẪN THÊM HIỆU ỨNG (cho người làm phần đồ họa)

Phần này dành cho bạn khi muốn gắn hiệu ứng mới cho một chiêu.

## Cần nhớ trước

Model đã nhận diện sẵn **6 chiêu** (index 0–5, xem `GESTURES` trong
`gesture_utils.py`). Gắn hiệu ứng cho các chiêu này **KHÔNG cần train lại** —
chỉ viết code Python. Bạn chỉ đụng tới ML khi muốn thêm một cử chỉ *hoàn toàn mới*
(xem mục cuối).

Danh sách chiêu hiện có:

| index | Cử chỉ | Hiệu ứng hiện tại |
|-------|--------|-------------------|
| 0 | Xòe bàn tay | Cầu lửa ✅ |
| 1 | Nắm đấm | *(chưa có)* |
| 2 | Ngón giữa | *(chưa có)* |
| 3 | Ngón trỏ (hi) | *(chưa có)* |
| 4 | Ngón cái | *(chưa có)* |
| 5 | Ngón trỏ + út | *(chưa có)* |

## Quy trình 3 bước

### Bước 1 — Viết class hiệu ứng trong `Fireball3D.py`

Mỗi hiệu ứng là một class kế thừa `Entity`, có `__init__` (dựng hình) và
`update()` (chạy mỗi frame — di chuyển, mờ dần, rồi `destroy(self)`). Xem
`Fireball`, `Ember`, `Smoke` làm mẫu. Có thể tái dùng `fire_shader`, `fire_tex`,
và các particle có sẵn.

Khung tối thiểu:

```python
class MyEffect(Entity):
    def __init__(self, position):
        super().__init__(model="sphere", position=position, unlit=True,
                         color=color.cyan)
        self.life = 1.0

    def update(self):
        self.life -= utime.dt
        self.alpha = max(0.0, self.life)     # mờ dần
        if self.life <= 0:
            destroy(self)                     # QUAN TRỌNG: dọn dẹp khi hết
```

### Bước 2 — Import + viết hàm `cast_...` trong `main.py`

Thêm class vừa viết vào dòng import, rồi viết hàm tạo hiệu ứng từ landmark.
Dùng `hand_to_world(lm[9].x, lm[9].y)` để lấy vị trí lòng bàn tay trong không
gian 3D (landmark 9 = giữa lòng bàn tay; 0 = cổ tay; 12 = đầu ngón giữa).

```python
from Fireball3D import bloom_shader, init_effects, Fireball, MyEffect

def cast_my_skill(lm):
    pos = hand_to_world(lm[9].x, lm[9].y)
    MyEffect(pos)
```

### Bước 3 — Gắn vào `update()`

Trong `main.py`, tại chỗ đánh dấu `>>> GAN CHIEU MOI TAI DAY`, thêm nhánh
`elif` cho số chiêu tương ứng. Nên cho mỗi chiêu một **cooldown riêng**.

```python
# khai báo cooldown gần _last_fire_time
_last_skill_time = 0.0
SKILL_COOLDOWN = 0.8

# trong update(), nhớ thêm biến vào dòng `global` ở đầu hàm:
def update():
    global _last_fire_time, _last_skill_time
    ...
    if gesture == 0 and ...:
        cast_fireball(lm); ...
    elif gesture == 1 and (time.time() - _last_skill_time > SKILL_COOLDOWN):
        cast_my_skill(lm)
        _last_skill_time = time.time()
```

## Ví dụ hoàn chỉnh — Chiêu 2 "Nắm đấm" = nổ tàn lửa tỏa tròn

Tái dùng luôn `Ember`, không cần viết class mới:

```python
# main.py
from Fireball3D import bloom_shader, init_effects, Fireball, Ember

_last_nova_time = 0.0
NOVA_COOLDOWN = 0.8

def cast_nova(lm):
    pos = hand_to_world(lm[9].x, lm[9].y)
    for i in range(24):
        a = i / 24 * 2 * math.pi
        e = Ember(pos)
        e.velocity = Vec3(math.cos(a) * 8, math.sin(a) * 8, -2)  # tỏa tròn
        e.life = 0.9

# trong update(): global _last_fire_time, _last_nova_time
    elif gesture == 1 and (time.time() - _last_nova_time > NOVA_COOLDOWN):
        cast_nova(lm)
        _last_nova_time = time.time()
```

## Mẹo & lưu ý

- **Tọa độ:** trục z âm = bay về phía màn hình (càng gần camera càng to nhờ phối
  cảnh). `SPAWN_Z = 16` là mặt phẳng xuất hiện; hiệu ứng bị `destroy` khi `z < 0.8`.
- **Glow:** muốn hiệu ứng phát sáng thì cho màu thật sáng (gần trắng) — bloom chỉ
  bắt sáng phần vượt `threshold`. Nền webcam đã bị giảm sáng (`BG_BRIGHTNESS`) nên
  không bị glow.
- **Texture khói/lửa:** dùng biến `fire_tex` trong `Fireball3D.py`, nhưng chỉ sau
  khi `init_effects()` đã chạy (main.py gọi sẵn sau `Ursina()`).
- **Luôn `destroy(self)`** khi hiệu ứng hết đời, nếu không số entity sẽ phình lên
  và tụt FPS (góc trên phải cửa sổ có hiện `entities:`).

---

## Khi nào PHẢI train lại model

Chỉ khi **thêm một cử chỉ mới chưa có** trong `GESTURES`, hoặc model nhận nhầm
một chiêu và bạn muốn thu thêm mẫu:

1. Sửa danh sách `GESTURES` trong `gesture_utils.py` (tối đa 10 chiêu, ứng phím 0–9).
2. `python collect_data.py` — giơ cử chỉ, **giữ phím số** tương ứng để ghi mẫu
   (~200–300 mẫu/chiêu, đổi góc/khoảng cách/tay trái–phải). Data ghi nối tiếp vào
   `gestures.csv`.
3. `python train_model.py` — train lại, xuất `gesture_model.joblib`.
4. Chạy lại `main.py`.

## Tham số tinh chỉnh nhanh

| Tham số | File | Ý nghĩa |
|---------|------|---------|
| `CONF_THRESHOLD` (0.8) | `main.py` | Dưới ngưỡng này coi như không rõ chiêu. |
| `SMOOTH_FRAMES` (7) | `main.py` | Số frame để bình chọn (làm mượt, chống nhấp nháy). |
| `FIRE_COOLDOWN` (0.15) | `main.py` | Nhịp bắn cầu lửa. |
| `threshold / intensity / blur_size` | `main.py` | Thông số bloom (glow). |
| `BG_BRIGHTNESS` (0.85) | `main.py` | Độ sáng webcam (phải < `threshold` để nền không bị glow). |

## Ghi chú repo

- `gestures.csv` và `gesture_model.joblib` là file sinh ra. Muốn repo gọn có thể
  không commit `.joblib` (người khác clone về tự `train_model.py`).
- `.vscode/` chứa đường dẫn `.venv` theo máy cá nhân — cân nhắc thêm vào `.gitignore`.