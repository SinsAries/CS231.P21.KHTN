# Hướng dẫn Chạy và Kiểm tra Mô hình Vision Mamba trên tập dữ liệu CIFAR-10

Tài liệu này cung cấp hướng dẫn chi tiết để cài đặt môi trường, chạy và kiểm tra mô hình Vision Mamba (`Vim`) đã được huấn luyện. Mục tiêu chính là hướng dẫn cách sử dụng các file checkpoint có sẵn để đánh giá hiệu suất của mô hình hoặc tiếp tục quá trình huấn luyện.

## Mục lục
1. [Yêu cầu về môi trường](#1-yêu-cầu-về-môi-trường)
2. [Cấu trúc thư mục](#2-cấu-trúc-thư-mục)
3. [Hướng dẫn thực thi](#3-hướng-dẫn-thực-thi)
    - [Bước 1: Cài đặt các thư viện cần thiết](#bước-1-cài-đặt-các-thư-viện-cần-thiết)
    - [Bước 2: Tải và thiết lập Checkpoint](#bước-2-tải-và-thiết-lập-checkpoint)
    - [Bước 3: Chạy Notebook và Đánh giá mô hình](#bước-3-chạy-notebook-và-đánh-giá-mô-hình)
4. [Giải thích về các file Checkpoint](#4-giải-thích-về-các-file-checkpoint)

## 1. Yêu cầu về môi trường
Mã nguồn được viết bằng Python và sử dụng framework PyTorch. Cần có GPU (khuyến khích NVIDIA GPU với CUDA) để chạy mô hình hiệu quả.

Các thư viện chính bao gồm:
- `torch` & `torchvision`
- `zetascale`
- `swarms`
- `einops`
- `tqdm`
- `matplotlib`
- `scikit-learn`
- `seaborn`

Tất cả các thư viện này sẽ được cài đặt tự động thông qua file notebook.

## 2. Cấu trúc thư mục
Để thuận tiện cho việc quản lý, vui lòng tổ chức các file theo cấu trúc sau:

```
/vision-mamba-cifar10/
|
|-- checkpoints/
| |-- model_checkpoint_epoch_119.pth <-- Ví dụ checkpoint
| |-- model_best_accuracy.pth <-- Ví dụ checkpoint tốt nhất
|-- ... (các file checkpoint khác)
|-- visionmambademo.ipynb <-- File mã nguồn chính
-- README.md <-- File hướng dẫn này
```

- **`checkpoints/`**: Thư mục này dùng để chứa tất cả các file checkpoint (`.pth`) mà em đã cung cấp.
- **`visionmambademo.ipynb`**: File Jupyter Notebook chứa toàn bộ mã nguồn để huấn luyện và đánh giá mô hình.

## 3. Hướng dẫn thực thi

### Bước 1: Cài đặt các thư viện cần thiết
1.  Mở file `visionmambademo.ipynb` bằng Jupyter Notebook, Google Colab, hoặc VS Code.
2.  Chạy ô code đầu tiên (cell 1) để cài đặt các thư viện `zetascale` và `swarms`. Quá trình này có thể mất vài phút.
    ```python
    !pip install zetascale
    !pip install swarms
    ```

### Bước 2: Tải và thiết lập Checkpoint
1.  Tạo một thư mục có tên là `checkpoints` cùng cấp với file `visionmambademo.ipynb`.
2.  Sao chép tất cả các file checkpoint (`.pth`) mà em đã gửi vào trong thư mục `checkpoints` này.

### Bước 3: Chạy Notebook và Đánh giá mô hình
Đây là bước quan trọng nhất. Thầy chỉ cần **chỉnh sửa một dòng mã** để chọn checkpoint muốn kiểm tra.

1.  Trong file `visionmambademo.ipynb`, tìm đến **ô code thứ 10**. Ô này chứa đoạn mã để khởi tạo các tham số và tải checkpoint.

2.  **Chỉnh sửa đường dẫn checkpoint:**
    Tìm dòng `prev_checkpoint = "..."`. Thầy hãy thay đổi đường dẫn trong dấu ngoặc kép để trỏ đến file checkpoint mong muốn trong thư mục `checkpoints`.

    **Ví dụ:**
    Nếu thầy muốn kiểm tra file checkpoint có tên là `model_best_accuracy.pth`, hãy sửa dòng đó thành:
    ```python
    # Dòng code gốc có thể là:
    # prev_checkpoint = "/kaggle/input/visionmambademo13/pytorch/default/1/model_checkpoint.pth"

    # SỬA THÀNH:
    prev_checkpoint = "checkpoints/model_best_accuracy.pth"
    ```
    Hoặc nếu muốn kiểm tra một checkpoint ở epoch cụ thể, ví dụ `model_checkpoint_epoch_119.pth`:
    ```python
    # SỬA THÀNH:
    prev_checkpoint = "checkpoints/model_checkpoint_epoch_119.pth"
    ```

3.  **Thực thi các ô code:**
    Sau khi đã chỉnh sửa đường dẫn, thầy thực hiện theo một trong hai kịch bản sau:

    **Kịch bản A: Chỉ Đánh giá kết quả (Không huấn luyện lại)**
    Đây là kịch bản nhanh nhất để xem hiệu suất của mô hình từ một checkpoint.
    - Chạy tuần tự các ô code từ đầu cho đến **ô code thứ 15** (ô `print(VisionMamba_model)`).
    - **BỎ QUA (KHÔNG CHẠY)** ô code thứ 16, là ô gọi hàm `train_model(...)`.
    - Chạy các ô code còn lại từ **ô 17 đến cuối** để xem các kết quả đánh giá:
        - Độ chính xác tổng thể trên tập test.
        - Ví dụ dự đoán trên ảnh ngẫu nhiên.
        - Ma trận nhầm lẫn (Confusion Matrix).
        - Báo cáo phân loại chi tiết (Precision, Recall, F1-score).

    **Kịch bản B: Tiếp tục Huấn luyện (Resume Training)**
    Nếu thầy muốn tiếp tục huấn luyện mô hình từ điểm checkpoint đã chọn:
    - Chạy tuần tự **tất cả các ô code** trong notebook từ đầu đến cuối. Quá trình huấn luyện sẽ được tiếp tục từ epoch đã lưu trong file checkpoint.

## 4. Giải thích về các file Checkpoint
Em đã cung cấp một số file checkpoint, thường bao gồm:
- **`model_checkpoint.pth`**: File checkpoint được lưu lại ở epoch gần nhất trong quá trình huấn luyện.
- **`model_bestcheckpoint.pth`**: File checkpoint lưu lại trọng số của mô hình tại epoch có độ chính xác (validation accuracy) cao nhất. **Đây là file được khuyến khích sử dụng để đánh giá hiệu suất cuối cùng của mô hình.**
- Các file có tên theo epoch (ví dụ: `checkpoint_epoch_119.pth`): Checkpoint được lưu tại một epoch cụ thể để tiện theo dõi.

---
Hy vọng tài liệu này sẽ giúp thầy dễ dàng chạy và kiểm tra mô hình. Nếu có bất kỳ thắc mắc nào, xin thầy cứ liên hệ với em.

