## Scripts

Thư mục `scripts/` chứa các entry-point (EDA → train → evaluate → báo cáo). Mỗi script chỉ điều phối, còn logic nằm trong thư viện `src/hr_employee`.

Luồng dữ liệu và thư mục
- `data/Churn_Modelling.csv`: dữ liệu gốc được đọc trực tiếp (hiện chưa tách thành processed/final). Nếu muốn sạch hơn, có thể tách:
	- `data/raw/`: dữ liệu gốc, chỉ đọc.
	- `data/processed/`: sau khi làm sạch cơ bản.
	- `data/final/`: sau feature engineering, sẵn sàng train.
- `outputs/`: kết quả số (JSON/CSV, model `.joblib`, bảng OR, summary statsmodels).
- `figures/`: biểu đồ sinh tự động (ROC, confusion matrix, churn rate theo nhóm, phân phối biến số).

Các entry-point chính (đã tinh gọn)
- `run_pipeline.py`: chạy end-to-end (EDA → train → evaluate → odds ratio → sinh báo cáo).
- `generate_project_report.py`: sinh `PROJECT_REPORT.md` từ artifacts hiện có (không train lại).

Cách chạy chuẩn
```bash
python scripts/run_pipeline.py
```
Yêu cầu: đã `pip install -e .` để import được `hr_employee`.


