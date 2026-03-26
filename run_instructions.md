# Hướng dẫn chạy dự án Thống kê Phân tích (HR Employee & HR Attrition)

Để chạy dự án này trên Windows một cách mượt mà và trơn tru nhất (từ đầu), bạn hãy thực hiện theo các bước sau trong Terminal (PowerShell hoặc CMD) tại thư mục `f:\data-statistical-analysis\`.

### Bước 1: Dọn dẹp kết quả cũ (Tùy chọn)
Chương trình có cơ chế tự động ghi đè, nhưng để chắc chắn bạn đang chạy lại từ một môi trường sạch, bạn có thể **xóa** các folder/file kết quả từ những lần chạy trước:
- Xóa thư mục `figures/`
- Xóa thư mục `outputs/`
- Xóa file `PROJECT_REPORT.md`
> **Phạt cảnh cáo**: Tuyệt đối **KHÔNG XÓA** thư mục `data/` (chứa file CSV) và thư mục `src/`, `scripts/`.

### Bước 2: Khởi tạo phân vùng môi trường Python (Virtual Environment)
Mở Terminal tại thư mục gốc của dự án `f:\data-statistical-analysis\`.

Tạo môi trường ảo (để cách ly thư viện, không sợ đụng độ với project khác):
```powershell
python -m venv venv
```

Kích hoạt môi trường vừa tạo:
```powershell
source .venv/bin/activate
```
*(Nếu thành công, bạn sẽ thấy chữ `(venv)` hiện lên ở đầu dòng lệnh Terminal).*

### Bước 3: Cài đặt Thư viện
Khi đã ở trong môi trường ảo, hãy cài đặt các thư viện lõi (scikit-learn, statsmodels, pandas...) bằng lệnh sau:
```powershell
pip install -r requirements.txt
```

> **Lưu ý**: Khác với một số code trên mạng bắt buộc phải gõ `pip install -e .`, dự án này đã được tối ưu đường dẫn. Bạn **BỎ QUA** lệnh setup `pip install -e .` luôn nhé! Hàm trong file `run_pipeline.py` sẽ tự động hiểu thư mục `src`.

### Bước 4: Chạy dự án (Run Pipeline)
Bây giờ mọi thứ đã sẵn sàng. Gõ lệnh sau để ra lệnh cho Python bắt đầu chạy quá trình phân tích dữ liệu, huấn luyện mô hình và tính Toán thống kê (Odds Ratio/P-value):
```powershell
python scripts/run_pipeline.py
```

### Bước 5: Xem thành quả 🚀
Bạn đợi khoảng 2-5 giây để Terminal chạy xong và báo `Pipeline completed successfully!`. 

Khi quay lại thư mục dự án, bạn sẽ nhận được:
1. Thư mục `figures/` chứa toàn bộ biểu đồ EDA (chuông phân phối, Random Forest plot về tỷ số Odds).
2. Thư mục `outputs/` chứa các bảng CSV Odds Ratio (tỷ số lệch odds), file tỷ lệ chính xác JSON và mô hình AI (.joblib) đã được lưu lại để có thể tái sử dụng.
3. File báo cáo tổng hợp chuẩn tự động sinh ra: `PROJECT_REPORT.md` để bạn sao chép thẳng vào Luận văn!
