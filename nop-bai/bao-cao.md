# Báo Cáo Lab Day 21 - CI/CD cho AI Systems

| Thông tin | Nội dung |
|---|---|
| Họ và tên | Bùi Thu Trang |
| MSSV | 2A202601758 |
| Lớp / Khóa | K4 |
| Repo GitHub | [Track2-Day21-2A202601758-BuiThuTrang](https://github.com/buithutrang90412/Track2-Day21-2A202601758-BuiThuTrang) |
| Ngày nộp | 21/08/2026 |

## 1. Bộ siêu tham số đã chọn và lý do

| Lần chạy | n_estimators | learning_rate | max_depth | f1_score | accuracy |
|---:|---:|---:|---:|---:|---:|
| 1 | 1000 | 0.001 | 5 | 0.6256 | 0.854 |
| 2 | 800 | 0.01 | 5 | 0.7196 | 0.880 |
| 3 | 600 | 0.01 | 5 | 0.7256 | 0.882 |

Bộ được chọn là `n_estimators=600`, `learning_rate=0.01`, `max_depth=5` vì đạt F1 cao nhất và vượt ngưỡng 0.65. Learning rate thấp cần nhiều cây hơn để bù lại mức đóng góp nhỏ của mỗi cây; nếu không tăng số cây phù hợp, F1 có thể giảm.

## 2. Vì sao dùng F1 thay vì accuracy

Tập dữ liệu mất cân bằng, chỉ 24,8% mẫu thuộc lớp thu nhập cao. Mô hình luôn dự đoán “thu nhập thấp” vẫn đạt accuracy khoảng 0,752 nhưng không nhận diện được mẫu thu nhập cao nào. F1-score kết hợp precision và recall của lớp dương, nên phản ánh khả năng phát hiện nhóm này tốt hơn accuracy. Lab dùng `f1_score(y_eval, preds)` mặc định cho lớp dương, không dùng `average="weighted"` hoặc `average="macro"` vì lớp đa số có thể kéo kết quả lên cao. Quality gate vì vậy đặt trên F1 với ngưỡng 0.65.

## 3. Khó khăn và cách giải quyết

| Khó khăn | Cách giải quyết |
|---|---|
| DVC báo thiếu plugin hoặc bị từ chối quyền S3 | Cài `dvc[s3]` và cấp quyền đọc/ghi object cho IAM user. |
| GitHub Actions không SSH được vào EC2 | Cập nhật Public IP, SSH key và Security Group cho port 22. |
| EC2 không tải được model do lệch phiên bản thư viện | Cài `scikit-learn==1.4.2` giống môi trường huấn luyện rồi restart service. |

## 4. So sánh Bước 2 và Bước 3

| Phiên bản | f1_score | accuracy |
|---|---:|---:|
| Bước 2: 22.361 mẫu | 0.7256 | 0.882 |
| Bước 3: 44.722 mẫu | 0.7442 | 0.890 |

Khi bổ sung 22.361 mẫu, F1 tăng 0.0186 và accuracy tăng 0.008. Dữ liệu mới cùng nguồn nhưng giúp mô hình học ổn định hơn trong lần chạy này; pipeline đã tự động pull dữ liệu, huấn luyện, kiểm tra quality gate và triển khai lại.

## 5. Bonus đã thực hiện

- [ ] Bonus 1: DagsHub chưa xác nhận run thực tế.
- [x] Bonus 2: Quét threshold 0.10–0.90 và chọn F1 cao nhất.
- [x] Bonus 3: Tạo confusion matrix, precision/recall trong `outputs/detail.txt`.
- [x] Bonus 4: Dừng release nếu F1 model mới thấp hơn model hiện tại trên S3.
- [x] Bonus 5: Cảnh báo khi tỷ lệ target dương lệch quá 5% so với 24,8%.
