# Hướng dẫn sử dụng Kaggle Notebook: Demo Mô hình Vision Mamba (Vim)

## 1. Tổng quan

Kính gửi thầy,

Notebook này là bản demo cho mô hình **Vision Mamba (Vim)** em đã huấn luyện để thực hiện bài toán phân loại ảnh trên bộ dữ liệu CIFAR-10 (gồm 10 lớp: máy bay, ô tô, chim, mèo, hươu, chó, ếch, ngựa, tàu, xe tải).

Mục tiêu của notebook là trình bày cách tải lại (load) một mô hình đã được huấn luyện từ file checkpoint và sử dụng nó để dự đoán trên một ảnh bất kỳ từ một đường dẫn URL.

## 2. Yêu cầu

Để chạy được notebook này trên Kaggle, cần đảm bảo các yếu tố sau:

1.  **Tài khoản Kaggle**: Cần đăng nhập vào Kaggle để có thể chạy notebook.
2.  **Dataset chứa Checkpoint**: Đây là bước quan trọng nhất. Mô hình đã huấn luyện được lưu trong một file checkpoint (`model_bestcheckpoint.pth`). File này cần được thêm vào notebook dưới dạng một Kaggle Dataset.
    * **Tên dataset cần thêm**: `vim-checkpoint`
    * **Đường dẫn truy cập trong notebook**: `/kaggle/input/vim-checkpoint/`

## 3. Cách sử dụng

Thầy có thể thực hiện theo các bước đơn giản sau:

1.  **Mở Notebook**: Truy cập vào link notebook trên Kaggle.
2.  **Thêm dữ liệu**:
    * Ở thanh công cụ bên phải của trình chỉnh sửa code, chọn mục **"Add Data"**.
    * Trong ô tìm kiếm, gõ `vim-checkpoint` và nhấn Enter.
    * Khi dataset hiện ra, nhấn nút **"Add"**. Kaggle sẽ tự động thêm dataset vào với đúng đường dẫn `/kaggle/input/vim-checkpoint/`.
3.  **Chạy Notebook**:
    * Cách đơn giản nhất là chọn **"Run All"** từ menu **"Run"** trên thanh công cụ.
    * Notebook sẽ tự động thực hiện các tác vụ theo thứ tự:
        * Cài đặt các thư viện cần thiết (`zetascale`, `swarms`).
        * Định nghĩa lại kiến trúc mô hình `Vim`.
        * Tải trọng số đã huấn luyện từ file checkpoint.
        * Chạy dự đoán trên một ảnh mẫu (một chiếc máy bay).

## 4. Tùy chỉnh và Thử nghiệm

Để thử nghiệm mô hình với một ảnh khác, thầy chỉ cần thay đổi đường dẫn URL trong ô code cuối cùng:

```python
# Thay đổi URL trong cặp dấu ngoặc kép bên dưới
url_plane = "[https://media.gq.com/photos/6508829d305ef4e0229049b3/master/w_2240,c_limit/plane.jpg](https://media.gq.com/photos/6508829d305ef4e0229049b3/master/w_2240,c_limit/plane.jpg)"

# Chạy hàm dự đoán với URL mới
predict_from_url(model, url_plane, device, classes)
```

Sau khi thay đổi URL, thầy chỉ cần chạy lại ô code đó bằng cách nhấn vào nút ▶️ (Run) ở bên cạnh ô. Kết quả dự đoán và hình ảnh sẽ được hiển thị ngay bên dưới.

Em xin chân thành cảm ơn thầy.

https://www.kaggle.com/code/thnhnguyntrngtt/demo-vim
