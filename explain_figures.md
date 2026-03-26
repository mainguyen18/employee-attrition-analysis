# Giải thích chi tiết các biểu đồ phân tích nhân sự (HR Attrition)

Sau khi chạy lệnh `python scripts/run_pipeline.py`, mã nguồn đã thực hiện ba quá trình chính: Xử lý dữ liệu, Huấn luyện mô hình học máy (Logistic Regression) để dự đoán khả năng nghỉ việc (Churn) và Đánh giá thống kê. Các file hình ảnh sinh ra trong thư mục `figures/` là kết quả trực quan hóa của quá trình này. Dưới đây là giải thích chi tiết cho từng ảnh một.

---

## Nhóm 1: Tỷ lệ nghỉ việc theo các biến danh mục (Categorical - Churn Rate)

Các biểu đồ này (thường là biểu đồ cột/Bar chart) dùng để trả lời câu hỏi: *"Nhân viên mang đặc điểm nào thì có khả năng nghỉ việc cao hơn?"*

1. **`churn_rate_by_BusinessTravel.png`**
   - **Nội dung:** Tỷ lệ nghỉ việc dựa trên mức độ đi công tác của nhân viên (Không bao giờ, Thỉnh thoảng, hay Thường xuyên).
   - **Ý nghĩa:** Chắc chắn bạn sẽ thấy nhóm hay đi công tác tỷ lệ nghỉ việc cao hơn do áp lực di chuyển và thiếu cân bằng công việc - cuộc sống.

2. **`churn_rate_by_EducationField.png`**
   - **Nội dung:** Tỷ lệ nghỉ việc theo chuyên ngành đào tạo (Ví dụ: Y khoa, Kỹ thuật, Nhân sự, Marketing...).
   - **Ý nghĩa:** Giúp thấy được sự dịch chuyển lao động đặc thù ở nhóm ngành nào đang diễn ra mạnh nhất.

3. **`churn_rate_by_Gender.png`**
   - **Nội dung:** Tỷ lệ nghỉ việc có sự khác biệt giữa Nam và Nữ hay không.
   - **Ý nghĩa:** Phân tích yếu tố giới tính ảnh hưởng đến quyết định gắn bó lâu dài.

4. **`churn_rate_by_JobRole.png`**
   - **Nội dung:** Thống kê tỷ lệ nghỉ việc theo từng vị trí (Nhân viên kinh doanh, Quản lý, Kỹ sư, Giám đốc, v.v.).
   - **Ý nghĩa:** Bạn có thể nhìn vào đây để xem phòng ban/vị trí nào công ty đang bị "chảy máu chất xám" nặng nhất để kịp thời can thiệp.

5. **`churn_rate_by_MaritalStatus.png`**
   - **Nội dung:** Khả năng nghỉ việc theo tình trạng hôn nhân (Độc thân, Đã kết hôn, hay Đã ly hôn).
   - **Ý nghĩa:** Thường thì nhân viên độc thân ít vướng bận gia đình nên họ có xu hướng luân chuyển việc làm nhiều hơn.

6. **`churn_rate_by_OverTime.png`**
   - **Nội dung:** So sánh tỷ lệ nghỉ việc giữa nhóm nhân viên có làm thêm giờ (OverTime = Yes) và không làm thêm giờ.
   - **Ý nghĩa:** Đây thường là yếu tố rực đỏ sinh ra rủi ro nghỉ việc rất cao do nhân viên bị kiệt sức (burnout).

---

## Nhóm 2: Phân bố của các thông số định lượng (Numeric Distributions)

Các hình có tiền tố `dist_` so sánh trực quan các yếu tố bằng con số qua 2 nhóm: Người đã nghỉ việc (Đường màu Cam/Đỏ) và Người ở lại (Đường màu Xanh).

> **💡 GIẢI THÍCH CÁCH ĐỌC TRỤC DỌC (TRỤC Y) CỦA NHÓM 2:**
> Rất nhiều người bối rối với con số ở cột dọc. Có 2 trường hợp hiển thị:
> - **Trường hợp 1 (Nếu là số nguyên lớn như 0, 50, 100...):** Trục Y mang ý nghĩa là **Số lượng người (Count)**. Ví dụ: cột ngang mức 100 nghĩa là có đúng 100 nhân viên đang ở đỉnh biểu đồ.
> - **Trường hợp 2 (Nếu là các số lẻ cực nhỏ như 0.01, 0.05...):** Đây là *Biểu đồ Mật độ (KDE Density)*. Trục Y mang ý nghĩa là **Mật độ tập trung**. Tổng diện tích phía dưới đường cong luôn là 1 (100%). Con số lẻ `0.06` không phải là 6%, mà nó biểu thị mức độ "trầm trọng, đậm đặc" của nhân sự hội tụ tại điểm đó. BẠN KHÔNG CẦN NHÌN CON SỐ NÀY! Bạn chỉ quan tâm **đỉnh đường cong nào nhô cao nhất** và hình dáng của nó nằm lệch về cột mốc nào để kết luận.

1. **`dist_Age_by_churn.png` (Độ phân bố Tuổi tác)**
   - **Hình dáng hiển thị:** Đỉnh quả đồi của nhóm *Đã nghỉ việc* thường nằm ở mốc 28-32 tuổi. Trong khi đỉnh nhóm *Ở lại* thường lệch dịch sang phải ở mốc 35-40 tuổi.
   - **Kết luận:** Độ tuổi càng trẻ, khả năng nhảy việc càng cao. Người trên 35 tuổi có xu hướng tìm kiếm sự an toàn, trung thành hơn hẳn.

2. **`dist_DistanceFromHome_by_churn.png` (Khoảng cách đi làm)**
   - **Hình dáng hiển thị:** Đường cong của nhóm *Nghỉ việc* có một chỗ phình to hơn ở khoảng cách xa (10km đến 25km).
   - **Kết luận:** Quãng đường đi làm xa thực sự tạo ra hệ lụy khổng lồ. Khoảng cách trung bình nhóm nghỉ là 10.6km so với nhóm ở lại là 8.9km. Đi làm quá xa vắt kiệt thể lực nhân sự trước cả khi họ vào tới văn phòng.

3. **`dist_MonthlyIncome_by_churn.png` (Phân bố Thu nhập)**
   - **Hình dáng hiển thị:** Nhóm *Nghỉ việc* có một cái "đỉnh" cực kì nhọn và vút cao ở vùng lương thấp ($2,000 - $3,000/tháng). Ngược lại nhóm *Ở lại* có đường cong trải đều về phía mức lương cao ($5000 - $10,000).
   - **Kết luận:** Lương là nút thắt tử huyệt. Mật độ nghỉ việc tụ tập dày đặc ở ranh giới dưới thụ hưởng. Khi vượt qua ngưỡng thu nhập an toàn (~$5000), tỷ lệ chủ động từ bỏ công việc tuột dốc không phanh.

4. **`dist_YearsAtCompany_by_churn.png` (Thâm niên)**
   - **Hình dáng hiển thị:** Trong mô hình, những người *Nghỉ việc* dồn thành một cực đông đúc rõ rệt ở mốc 0 - 2 năm đầu tiên. Những người *Ở lại* sẽ có đường cong chồi lên ở các mốc thâm niên 5, 7 và 10 năm.
   - **Kết luận:** Nhân viên tàng hình ngay trong "Giai đoạn trăng mật". Nếu qua khỏi thử thách 1-2 năm đầu tiền hòa nhập môi trường, thì khả năng giữ chân họ trên 5 năm là hoàn toàn khả thi.

---

## Nhóm 3: Đánh giá chất lượng của mô hình (Model Evaluation & Statistics)

Nhóm hình này là kết tinh về toán học, minh họa cho nhận định: *"Con AI (Logistic Regression) vừa huấn luyện có khôn ngoan không, và nó tìm ra được quy luật gì?"*

1. **`confusion_matrix.png` (Ma trận Nhầm lẫn)**
   - **Mục đích:** Bảng điểm chấm thi cho mô hình AI, đọ sức "Dự đoán" so với "Thực tế".
   - **Ý nghĩa 4 ô vuông:**
     - **Góc trên-trái (True Negative):** Thực tế KHÔNG NGHỈ, máy đoán KHÔNG NGHỈ. (Tính chính xác).
     - **Góc trên-phải (False Positive):** Thực tế KHÔNG NGHỈ nhưng máy lại đoán là SẼ NGHỈ. (Tội báo động giả, đa nghi vô cớ).
     - **Góc dưới-trái (False Negative):** Thực tế ĐÃ NGHỈ nhưng máy lại đoán là KHÔNG NGHỈ. (Tội bỏ lọt lính đào ngũ, gây tổn thất bất ngờ).
     - **Góc dưới-phải (True Positive):** Thực tế ĐÃ NGHỈ và máy túm trúng phóc!
   - 👉 Mô hình càng xuất sắc thì đường chéo chính (Trên-trái & Dưới-phải) số càng khổng lồ, hai vị trí còn lại số càng hẻo lánh.

2. **`roc_curve.png` (Đường cong ROC - Đo lường Tầm Nhìn Mô Hình)**
   - **Mục đích:** Chứng minh thuật toán AI không phải là đứa nhắm mắt tự tung đồng xu đoán bừa.
   - **Ý nghĩa Trục & Đường nén:**
     - Trục dọc (TPR): Tỷ lệ lùng bắt thành công người nghỉ.
     - Trục ngang (FPR): Tỷ lệ bắt nhầm người vô tội.
     - Đường đứt khúc chéo: Đường của sự ngẫu nhiên 50-50.
   - **Ý nghĩa chỉ số AUC:** Nó thể hiện phần diện tích bên dưới cái đường cong dự đoán đó. 
     - Nếu `AUC = 0.5`: Vô dụng, máy dự đoán đoán mò.
     - Nếu `AUC = 1.0`: Máy thần thánh.
     - Giả sử `AUC = 0.85`: Máy có sức mạnh đến mức - cứ lôi ngẫu nhiên 1 người ở lại và 1 người nghỉ việc ném vào, máy có 85% năng lực xác định được chính xác ai là đứa sắp rời đi.

3. **`forest_plot_odds_ratio.png` (Biểu đồ Rừng - Đánh giá Tỷ suất Rủi ro)**
   - **Mục đích:** Bức tranh quan trọng nhất dùng để lên báo cáo định hướng chiến lược lãnh đạo. Nó chỉ mặt đặt tên thủ phạm trực tiếp thắt cổ nhân sự.
   - **Ý nghĩa hình ảnh:**
     - Có một ranh giới thẳng đứng tại mốc Toán học là **1.0 (Vạch Không Ảnh Hưởng)**.
     - Mỗi "Chấm vuông" là một tính chất (ví dụ Lương, Tăng ca...).
   - **Cách đọc Chấm vuông và Râu:**
     - Nằm tít về **BÊN PHẢI (Odds Ratio > 1.0):** Kẻ hủy diệt! Ai mang yếu tố này thì tỷ lệ nghỉ việc cao vọt gấp nhiều lần. (Ví dụ dấu chấm `OverTime` thường là 2.5 hoặc 3.0).
     - Nằm thụt về **BÊN TRÁI (Odds Ratio < 1.0):** Tấm khiên bảo mệnh! Yếu tố này giúp xoa dịu và giảm tỷ lệ nghỉ hưu xuống dưới mức thông thường (Ví dụ lương thực thụ cao làm OR xuống 0.6).
     - **Râu (Khoảng tin cậy 95%):** Chính là 2 nhánh gạch đâm ngang ra từ chấm vuông rủi ro. NẾU cái Râu mọc chọc xuyên qua vạch trung lập 1.0, chứng tỏ về phương diện thống kê sự dao động quá lớn, vô tình chạm mặt số 0, dẫn tới việc loại trừ hoàn toàn kết luận về yếu tố đó (không có giá trị báo cáo).
