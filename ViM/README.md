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
|-- CheckPoint/ <-- Thư mục này sẽ được tạo sau khi giải nén
| |-- 1/
| | |-- model_bestcheckpoint.pth
| |   -- model_checkpoint.pth
| |-- 2/
| |-- ... (cho đến thư mục 13)
| |-- 13/
| |-- model_bestcheckpoint.pth
| |-- model_checkpoint.pth
|-- visionmambademo.ipynb <-- File mã nguồn chính
-- README.md <-- File hướng dẫn này
```

- **`CheckPoint/`**: Thư mục chứa các thư mục con, mỗi thư mục con (1, 2, ..., 13) tương ứng với một phiên lưu checkpoint.
- **`visionmambademo.ipynb`**: File Jupyter Notebook chứa toàn bộ mã nguồn.

## 3. Hướng dẫn thực thi

### Bước 1: Cài đặt các thư viện cần thiết
1.  Mở file `visionmambademo.ipynb` bằng Jupyter Notebook, Google Colab, hoặc VS Code.
2.  Chạy ô code đầu tiên (cell 1) để cài đặt các thư viện `zetascale` và `swarms`. Quá trình này có thể mất vài phút.
    ```python
    !pip install zetascale
    !pip install swarms
    ```

### Bước 2: Tải và Giải nén Checkpoint
1.  **Tải file nén:** Truy cập đường dẫn Google Drive dưới đây để tải file `CheckPoint.rar` về máy.
    > **Link tải:** [drive](https://drive.google.com/file/d/1SjfkBgYFB7PkP-v3EYzXpuf7m8fHENXI/view?usp=sharing)

2.  **Giải nén:** Sử dụng phần mềm giải nén như WinRAR hoặc 7-Zip để giải nén file vừa tải. Quá trình giải nén sẽ tạo ra một thư mục tên là `CheckPoint`.

3.  **Đặt đúng vị trí:** Đảm bảo rằng thư mục `CheckPoint` vừa giải nén nằm **cùng cấp** (trong cùng một thư mục) với file `visionmambademo.ipynb`.

### Bước 3: Chạy Notebook và Đánh giá mô hình
Đây là bước quan trọng nhất. Thầy chỉ cần **chỉnh sửa một dòng mã** để chọn checkpoint muốn kiểm tra.

1.  Trong file `visionmambademo.ipynb`, tìm đến **ô code thứ 10**. Ô này chứa đoạn mã để khởi tạo các tham số và tải checkpoint.

2.  **Chỉnh sửa đường dẫn checkpoint:**
    Vì file nén chứa checkpoint của nhiều phiên huấn luyện (được đánh số từ 1 đến 13), bạn cần chỉ định rõ sẽ sử dụng bộ checkpoint từ thư mục nào.

    **Chúng tôi đề nghị sử dụng checkpoint từ lần huấn luyện cuối cùng, nằm trong thư mục `13`.**

    Trong file notebook, tìm dòng `prev_checkpoint = "..."` và thay thế bằng đường dẫn đầy đủ như ví dụ dưới đây.

    **Ví dụ:**
    Để sử dụng checkpoint **tốt nhất** từ thư mục `13`, hãy sửa dòng đó thành:
    ```python
    # SỬA THÀNH:
    prev_checkpoint = "CheckPoint/13/model_bestcheckpoint.pth"
    ```
    Để sử dụng checkpoint **thường** (lưu ở cuối epoch) từ thư mục `13`, hãy sửa thành:
     ```python
    # HOẶC SỬA THÀNH:
    prev_checkpoint = "CheckPoint/13/model_checkpoint.pth"
    ```
    *(Nếu muốn dùng một phiên bản khác, ví dụ từ thư mục `10`, bạn chỉ cần thay số `13` thành `10` trong đường dẫn.)*

3.  **Thực thi các ô code:**
    Sau khi đã chỉnh sửa đường dẫn, thầy thực hiện theo một trong hai kịch bản sau:

    **Kịch bản A: Chỉ Đánh giá kết quả (Không huấn luyện lại)**
    - Chạy tuần tự các ô code từ đầu cho đến **ô code thứ 15**.
    - **BỎ QUA (KHÔNG CHẠY)** ô code thứ 16 (ô `train_model(...)`).
    - Chạy các ô code còn lại từ **ô 17 đến cuối** để xem kết quả.

    **Kịch bản B: Tiếp tục Huấn luyện (Resume Training)**
    - Chạy tuần tự **tất cả các ô code** trong notebook từ đầu đến cuối.
