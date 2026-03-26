# Giải thích chi tiết các file dữ liệu trong thư mục `outputs`

Thư mục `outputs/` chính là "kho báu" chứa các giá trị cốt lõi nhất của dự án. Thay vì chỉ xuất ra hình ảnh (chỉ dùng để nhìn bằng mắt), `outputs/` lưu trữ **dữ liệu định lượng thô (con số)** và **bộ não của AI** để máy tính sử dụng cho các ứng dụng thực tế sau này (như tích hợp lên Website, Dashboard, hay đưa vào báo cáo tự động).

Dựa trên chức năng, 12 file trong thư mục này được phân loại thành 4 nhóm chính như sau:

---

## Nhóm 1: Dữ liệu bóc tách tỷ lệ nghỉ việc (6 file CSV)
**Gồm các file:** Nhóm `churn_rate_by_*.csv` (ví dụ: `BusinessTravel`, `EducationField`, `Gender`, `JobRole`, `MaritalStatus`, `OverTime`).

* **Ý nghĩa:** Đây chính là bản chất dạng "Bảng số liệu thô" của các biểu đồ Cột (Bar chart) trong Nhóm 1 ở thư mục `figures/`. 
* **Phân tích ứng dụng:** Thay vì chỉ cầm hình vẽ đi báo cáo chung chung, bộ phận nhân sự (HR) có thể import thẳng 6 file CSV này vào **Excel, PowerBI hay Tableau**. Trong file đếm cực kỳ chính xác dân số hiện tại: Ví dụ có bao nhiêu Kỹ sư IT, bao nhiêu Quản lý, cụ thể rụng bao nhiêu người và tính ra mẫu số là bao nhiêu %. Nó rất lý tưởng khi bạn cần đính kèm số liệu chính xác từng người đằng sau tấm biểu đồ để thuyết trình.

---

## Nhóm 2: Bảng tóm tắt Dữ liệu & Bảng điểm rèn luyện AI (2 file JSON)

### 1. `eda_summary.json`
* **Ý nghĩa:** Đây là "Sơ yếu lý lịch" nhanh gọn của bộ dữ liệu nguồn đầu vào.
* **Phân tích:** Nó chốt lại cấu trúc cơ sở: Dữ liệu hiện có bao nhiêu dòng (nhân viên), bao nhiêu cột tính chất, tỷ lệ "nhân viên nghỉ việc" chung của cả công ty là bao nhiêu % (thường ở cỡ ~16%). Đặc biệt, file này tự động rà soát xem có dòng/cột nào bị bỏ trống tính năng hay nhập thiếu dữ liệu (*missing values*) cần dọn dẹp hay không.

### 2. `logistic_metrics.json`
* **Ý nghĩa:** Đây đích thị là "Bảng điểm chấm thi" năng lực của con AI.
* **Phân tích:** Nó ghi lại rõ ràng từng phần trăm thập phân các chỉ số năng lực của não bộ máy tính: Độ chính xác (Accuracy), Độ nhạy (Recall), Điểm F1, Điểm phân hạng tổng quát (ROC AUC), và ghi nhớ kết quả 4 ô vuông của Ma trận nhầm lẫn lúc thi sát hạch trên tập dữ liệu Test. File này lưu lại sức mạnh mô hình, giúp lập trình viên đọ xem lần code update thuật toán của ngày hôm nay có khôn ngoan hơn ngày hôm qua hay không.

---

## Nhóm 3: Bộ não thao túng - Lõi Trí tuệ Nhân tạo (1 file Joblib)

### `logistic_pipeline.joblib` (Kích cỡ ~7KB)
* **Ý nghĩa:** TÀI SẢN QUAN TRỌNG NHẤT CỦA TOÀN DỰ ÁN. Đây là file nhị phân đóng băng toàn bộ sự "hiểu biết học thuật" mà máy học (Machine Learning) vừa nghiền ngẫm được.
* **Phân tích ứng dụng:** Việc huấn luyện mô hình tốn rất nhiều thời gian giải phương trình và tự tinh chỉnh trọng số mới tìm ra được quy luật ẩn. Do đó, việc nén hệ thống lại thành file `.joblib` mang ý nghĩa thực tiễn kinh doanh cốt lõi. \
**Ví dụ thực tế:** Giả sử tháng tới công ty tuyển thẳng 1,000 nhân viên mới. Chuyên gia Data Science sẽ KHÔNG phải chạy lại code học từ vạch xuất phát nữa. Họ chỉ viết một câu lệnh tí hon, load bộ não `.joblib` này lên, ném 1,000 CV nhân viên mới vào. Chưa đầy một tích tắc, não bộ sẽ "Xuất ra ngay danh sách các tân binh bị chấm điểm rủi ro có khả năng sẽ xin nghỉ việc/đào tẩu cao nhất trong tương lai". Cực kỳ uy lực!

---

## Nhóm 4: Bằng chứng luận điểm (Thống kê Statistical Inference & Odds Ratio)
Đây là nhóm 3 file làm cơ sở khoa học, chứng minh tường minh biến số/yếu tố nào gây nhảy việc, ở mức độ nào.

### 1. `odds_ratio_table.csv`
* **Ý nghĩa:** Bảng Excel tổng hợp các chỉ số cực đoan rủi ro (Odds Ratio) của từng hoàn cảnh cá nhân nhân sự đã được quy đổi.
* **Phân tích:** Số liệu này dùng trực tiếp để máy tính vẽ nên hình *Rừng Rủi ro (Forest Plot)* ở đầu chóp. File có cột định danh p-value và Khoảng giới hạn chuẩn (Confidence Interval) dùng làm bằng chứng pháp lý rành rọt bảo vệ việc bác bỏ đi hay giữ lại nguyên nhân đe dọa (Nếu p-value > 0.05, yếu tố đó lập tức bị gạch bỏ khỏi báo cáo).

### 2. `logit_summary.txt`
* **Ý nghĩa:** Báo cáo thuật toán thuần hàn lâm, khô khan.
* **Phân tích:** Tập tin này sinh ra đầy rẫy các số hạng của thống kê toán học truyền thống (Z-score, Pseudo R-squared, Log-Likelihood...). Nó sinh ra chỉ để phục vụ cho các trang báo cáo khoa học gò bó, hoặc in phụ lục đăng luận văn đại học. Khách hàng/sếp của bạn sẽ chẳng bao giờ muốn/cần đọc mớ bòng bong này.

### 3. `odds_ratio_analysis.txt` (Kích cỡ ~14KB)
* **Ý nghĩa:** TÀI LIỆU DỊCH THUẬT QUÝ GIÁ NHẤT CHO THỊ TRƯỜNG KINH DOANH.
* **Phân tích:** File này dùng lập trình (Auto-generated text) biến đóng ma trận số liệu rối não kia thành **những dòng ngôn ngữ tự nhiên**. Nó chỉ mặt đặt tên rành rọt sẵn xem "Yếu tố x, y, z có lợi hay có hại ra sao". Tài liệu này giống hệt như một bản nháp phân tích của chuyên viên tư vấn Nhân sự cấp cao. Công việc của bạn chỉ đơn giản là: Copy/Paste ý chính từ file `.txt` này thẳng đứng vào Slide báo cáo đệ trình Giám đốc nhân sự mà không tốn công giải thích số tính toán lằng nhằng!
