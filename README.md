# Elephant Movement Anomaly Detection  
## Data Preprocessing – Model Training – Data Quantization

## 1. Tổng quan

Mục tiêu của đề tài là phát hiện hành vi di chuyển bất thường của voi trong tự nhiên dựa trên dữ liệu GPS và các đặc trưng hành vi. Trong giai đoạn hiện tại, hệ thống tập trung vào ba bước chính:

1. Tiền xử lý dữ liệu và trích xuất đặc trưng
2. Huấn luyện mô hình Random Forest
3. Lượng tử hóa dữ liệu để chuẩn bị cho triển khai trên hệ thống tài nguyên hạn chế

Tài liệu này mô tả chi tiết pipeline từ dữ liệu đầu vào cho đến bước lượng tử hóa dữ liệu theo đúng quá trình triển khai trong mã nguồn.

---

## 2. Tiền xử lý dữ liệu và trích xuất đặc trưng

### 2.1 Dữ liệu đầu vào

Dữ liệu đầu vào được trích xuất từ dữ liệu GPS của voi, bao gồm:
- Thời gian (timestamp)
- Vĩ độ, kinh độ
- Các thông tin động học và thống kê liên quan

Sau quá trình tiền xử lý, dữ liệu được tổng hợp thành tập đặc trưng `elephant_features_kde_enhanced.csv`, trong đó mỗi dòng đại diện cho một cửa sổ thời gian.

---

### 2.2 Trích xuất đặc trưng hành vi

Các nhóm đặc trưng chính bao gồm:
- Đặc trưng động học: tốc độ, gia tốc
- Đặc trưng hình học quỹ đạo: bearing, turning angle, entropy
- Đặc trưng không gian: KDE probability, khoảng cách tới centroid
- Đặc trưng thống kê theo thời gian: mean, max, std, median
- Nhãn mục tiêu: `is_outside` (0: inside, 1: outside)

Nhãn `is_outside` được xác định dựa trên phân bố không gian KDE, phản ánh việc voi di chuyển ra ngoài vùng home range quen thuộc.

---

## 3. Huấn luyện mô hình học máy

### 3.1 Chuẩn bị dữ liệu

Tập dữ liệu đặc trưng được chia thành:
- Feature matrix X
- Nhãn mục tiêu y (`is_outside`)

Dữ liệu được chia thành tập huấn luyện và tập kiểm tra theo tỉ lệ 80/20, sử dụng stratified split để đảm bảo phân bố lớp.

---

### 3.2 Xử lý mất cân bằng dữ liệu

Do dữ liệu bất thường chiếm tỉ lệ nhỏ, pipeline xử lý mất cân bằng được áp dụng:
- SMOTE để tăng số lượng mẫu của lớp thiểu số
- Random Under Sampling để giảm số mẫu của lớp đa số

Quá trình này được tích hợp trong pipeline huấn luyện để tránh rò rỉ dữ liệu trong cross-validation.

---

### 3.3 Mô hình Random Forest

Mô hình Random Forest được sử dụng với các ưu điểm:
- Khả năng xử lý dữ liệu phi tuyến
- Ít nhạy cảm với thang đo của feature
- Phù hợp với dữ liệu đã được lượng tử hóa

Các siêu tham số được tối ưu bằng GridSearchCV với thước đo F1-macro.

---

### 3.4 Đánh giá và tối ưu threshold

Mô hình được đánh giá bằng:
- Confusion Matrix
- Precision, Recall, F1-score
- ROC-AUC

Ngoài ra, decision threshold được tối ưu bằng cách quét ngưỡng xác suất dự đoán nhằm tối đa hóa F1-score của lớp bất thường (`is_outside`).

---

## 4. Lượng tử hóa dữ liệu

### 4.1 Mục tiêu lượng tử hóa

Bước lượng tử hóa dữ liệu được thực hiện nhằm:
- Chuyển dữ liệu từ dạng số thực sang số nguyên
- Giảm độ phức tạp tính toán
- Chuẩn bị cho triển khai trên hệ thống nhúng hoặc phần cứng chuyên dụng

Trong giai đoạn này, chỉ thực hiện lượng tử hóa dữ liệu, chưa thực hiện lượng tử hóa mô hình.

---

### 4.2 Mã hóa các cột dạng chuỗi

Các cột có kiểu dữ liệu dạng chuỗi (object) được mã hóa bằng `LabelEncoder`.  
Mỗi giá trị chuỗi được ánh xạ sang một giá trị số nguyên tương ứng.

Thông tin ánh xạ nhãn được lưu lại dưới hai định dạng:
- `label_encoding_mapping.csv`: thuận tiện cho việc kiểm tra và báo cáo
- `label_encoding_mapping.json`: thuận tiện cho việc load và giải mã ngược trong chương trình

---

### 4.3 Phương pháp lượng tử hóa số

Các cột số (int64, float64) được lượng tử hóa độc lập bằng phương pháp min–max linear quantization.

Quy trình cho mỗi đặc trưng:
- Xác định giá trị nhỏ nhất (min) và lớn nhất (max)
- Ánh xạ tuyến tính về miền số nguyên không dấu 32-bit (uint32)
- Xử lý các giá trị không hợp lệ (NaN, ±∞)

Công thức lượng tử hóa:
quantized_value = round((x − min) / scale)

Trong đó:
scale = (max − min) / (2^32 − 1)

---

### 4.4 Bảng tham số lượng tử hóa

Với mỗi đặc trưng số, các tham số sau được lưu lại:
- Tên đặc trưng
- Scale
- Giá trị min
- Giá trị max

Bảng này được xuất ra file `Quantization_Scales.csv` để phục vụ cho:
- Giải lượng tử (dequantization)
- Đảm bảo tính nhất quán giữa train và inference
- Triển khai trên hệ thống phần cứng

---

### 4.5 Các file đầu ra của quá trình lượng tử hóa

Quá trình lượng tử hóa tạo ra các file sau:
- `Quantized_Combined_Features.csv`: tập dữ liệu đã được lượng tử hóa
- `Quantization_Scales.csv`: bảng tham số scale và min–max cho từng đặc trưng
- `label_encoding_mapping.csv`: bảng ánh xạ nhãn dạng CSV
- `label_encoding_mapping.json`: bảng ánh xạ nhãn dạng JSON

---

## 5. Kết luận giai đoạn

Ở giai đoạn hiện tại, hệ thống đã:
- Hoàn thiện pipeline tiền xử lý và trích xuất đặc trưng
- Huấn luyện và đánh giá mô hình Random Forest
- Thực hiện lượng tử hóa dữ liệu theo định dạng số nguyên 32-bit

Các bước tiếp theo sẽ tập trung vào:
- Lượng tử hóa mô hình
- Đánh giá ảnh hưởng của lượng tử hóa tới hiệu năng
- Chuẩn bị cho triển khai trên nền tảng phần cứng mục tiêu
