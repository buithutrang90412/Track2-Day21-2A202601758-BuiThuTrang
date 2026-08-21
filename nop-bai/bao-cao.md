# Báo Cáo Lab Day 21 - CI/CD cho AI Systems

<!--
HƯỚNG DẪN - đọc rồi XÓA TOÀN BỘ các khối chú thích này sau khi điền xong:

  - Giới hạn: KHÔNG QUÁ 1 TRANG A4, tương đương khoảng 450 - 550 từ nội dung.
  - Chỉ điền vào các chỗ ___ và các ô trong bảng. Không thêm mục mới.
  - Viết bằng câu hoàn chỉnh, không gạch đầu dòng cụt lủn.
  - Kiểm tra độ dài sau khi đã xóa hết chú thích:
        wc -w nop-bai/bao-cao.md
    và xem trước bản in bằng cách mở file trên GitHub rồi Ctrl+P / Cmd+P.
-->

|              |                                                                                                                                                     |
| ------------ | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| Họ và tên | Bùi Thu Trang                                                                                                                                      |
| MSSV         | 2A202601758                                                                                                                                         |
| Lớp / Khóa | K4                                                                                                                                                  |
| Repo GitHub  | [github.com/buithutrang90412/Track2-Day21-2A202601758-BuiThuTrang.git](https://github.com/buithutrang90412/Track2-Day21-2A202601758-BuiThuTrang.git) |
| Ngày nộp   | 21/08/2026                                                                                                                                          |

---

## 1. Bộ Siêu Tham Số Đã Chọn và Lý Do

<!-- Khoảng 120 - 150 từ. Điền kết quả thật từ MLflow UI ở Bước 1, tối thiểu 3 lần chạy. -->

| Lần chạy | n_estimators | learning_rate | max_depth | f1_score    | accuracy |
| ---------- | ------------ | ------------- | --------- | ----------- | -------- |
| 1          | 1000         | 0.001         | 5         | 0.625641025 | 0.854    |
| 2          | 800          | 0.01          | 5         | 0.719626168 | 0.880    |
| 3          | 600          | 0.01          | 5         | 0.725581395 | 0.882    |
| 4          | 600          | 0.03          | 5         | 0.703196347 | 0.870    |
| 5          | 600          | 0.05          | 5         | 0.705882352 | 0.870    |
| 6          | 500          | 0.05          | 5         | 0.711711711 | 0.872    |
| 7          | 400          | 0.05          | 5         | 0.714932126 | 0.874    |
| 8          | 300          | 0.05          | 5         | 0.706422018 | 0.872    |
| 9          | 300          | 0.05          | 2         | 0.692307692 | 0.872    |
| 10         | 200          | 0.05          | 5         | 0.703703703 | 0.872    |
| 11         | 200          | 0.01          | 5         | 0.669950738 | 0.866    |
| 12         | 200          | 0.1           | 5         | 0.714932126 | 0.874    |
| 13         | 50           | 0.05          | 2         | 0.605128205 | 0.846    |
| 14         | 100          | 0.1           | 3         | 0.710900473 | 0.878    |

**Bộ siêu tham số đã chọn:** `n_estimators=600`, `learning_rate=0.01`, `max_depth=5`.

**Lý do:** Bộ `n_estimators=600`, `learning_rate=0.01`, `max_depth=5` được chọn vì đạt F1 cao nhất trong các lần chạy được thể hiện trên MLflow (0.7256), đồng thời vượt ngưỡng chất lượng 0.65. Lần chạy này cũng có accuracy cao nhất trong ba lần được ghi lại. Kết quả cho thấy khi giảm learning rate, cần tăng n_estimators lên giá trị phù hợp để mô hình có đủ vòng boosting và đạt hiệu quả tốt hơn; nếu giảm learning rate mà không tăng n_estimatiors phù hợp thì f1 vẫn sẽ giảm; tuy nhiên accuracy không phản ánh đầy đủ khả năng nhận diện lớp thu nhập cao như F1.

<!--
Trả lời trong phần Lý do:
  - Vì sao bộ này tốt hơn các bộ còn lại (dựa trên f1_score, không phải accuracy)?
  - Lần chạy có accuracy cao nhất có trùng với lần có f1_score cao nhất không?
    Nếu không, điều đó nói lên điều gì?
  - Bạn quan sát thấy đánh đổi nào giữa n_estimators và learning_rate?
-->

---

## 2. Vì Sao Ngưỡng Chất Lượng Đặt Trên F1 Chứ Không Phải Accuracy

Vì tập dữ liệu bị mất cân bằng, chỉ **24,8%** mẫu thuộc lớp thu nhập cao (target = 1). Một mô hình luôn dự đoán “thu nhập thấp” vẫn đạt accuracy khoảng **0,752**, nhưng không phát hiện được bất kỳ trường hợp thu nhập cao nào, nên accuracy gây hiểu nhầm. F1-score của lớp dương kết hợp precision và recall, qua đó phản ánh tốt hơn khả năng nhận diện đúng nhóm thu nhập cao và hạn chế cả dự đoán dương sai lẫn bỏ sót. Vì vậy, lab dùng `f1_score(y_eval, preds)` với mặc định là lớp dương làm chỉ số chính. Không dùng `average="weighted"` hoặc `average="macro"` vì các cách tính này gộp kết quả của cả hai lớp; đặc biệt weighted F1 có thể bị lớp đa số kéo lên cao, khiến mô hình trông tốt dù vẫn bỏ sót nhiều mẫu thu nhập cao. Ngưỡng triển khai được đặt trên F1 để bảo đảm mô hình đạt chất lượng thực tế.

<!-- Khoảng 120 - 150 từ. -->

<!--
Cần nêu được:
  - Phân bố lớp của tập dữ liệu (tỷ lệ lớp thu nhập > 50K) và hệ quả của nó.
  - Accuracy của một mô hình luôn trả lời "thu nhập thấp" là bao nhiêu, vì sao con số
    đó gây hiểu nhầm.
  - F1 của lớp dương đo điều gì mà accuracy không đo được.
  - Vì sao KHÔNG dùng average="weighted" hay average="macro" khi gọi f1_score.
-->

## 3. Khó Khăn Gặp Phải và Cách Giải Quyết

<!-- Nêu 2 - 3 khó khăn thật, mỗi ô một câu ngắn. -->

| Khó khăn | Nguyên nhân | Cách giải quyết |
| ---------- | ------------- | ------------------ |
| DVC không đẩy được dữ liệu lên S3 | Thiếu plugin `dvc-s3` và quyền S3 chưa đầy đủ | Cài `dvc[s3]`, cấu hình remote S3 và cấp quyền đọc/ghi object cho IAM user. |
| GitHub Actions không SSH được vào EC2 | Security Group chỉ cho phép IP cá nhân truy cập port 22 | Mở tạm port 22 cho GitHub Actions, cập nhật đúng `SERVER_HOST` và cấu hình SSH key. |
| API không tải được model trên EC2 | Phiên bản `scikit-learn` trên EC2 khác phiên bản dùng để huấn luyện model | Cài đúng `scikit-learn==1.4.2`, khởi động lại systemd service và kiểm tra `/healthz`. |

---

## 4. So Sánh Bước 2 và Bước 3 (bắt buộc, 2 - 3 câu)

<!-- Lấy số liệu từ bảng ở mục 3.6 của tasks/buoc-3.md. -->

|                                  | f1_score | accuracy |
| -------------------------------- | -------- | -------- |
| Bước 2 (chỉ`train_batch1`)  | 0.7256   | 0.882    |
| Bước 3 (thêm`train_batch2`) | 0.7442   | 0.890    |

**Nhận xét:** Khi bổ sung thêm 22.361 mẫu cùng phân phối, f1_score tăng từ 0.7256 lên 0.7442 và accuracy tăng từ 0.882 lên 0.890. Điều này cho thấy dữ liệu mới giúp mô hình nhận diện lớp thu nhập cao tốt hơn trong lần chạy này, đồng thời pipeline đã tự động huấn luyện và triển khai lại thành công.

<!--
Một câu trả lời trung thực kiểu "f1 giảm 0,01 vì dữ liệu mới cùng phân phối, không mang
thêm thông tin mới" được đánh giá cao hơn kết luận sai rằng thêm dữ liệu luôn tốt hơn.
-->

---

## 5. Phần Bonus Đã Thực Hiện (nếu có)

<!-- Xóa cả mục 5 nếu không làm bonus. Mỗi bonus tối đa 1 dòng. -->

- [ ] Bonus 1 - Tracking MLflow từ xa với DagsHub: Chưa thực hiện vì cần cấu hình tài khoản DagsHub.
- [x] Bonus 2 - Điều chỉnh ngưỡng quyết định: Quét ngưỡng 0.10–0.90, chọn ngưỡng có F1 cao nhất và ghi vào report/MLflow.
- [x] Bonus 3 - Báo cáo precision / recall tự động: Tạo confusion matrix cùng precision/recall từng lớp trong `outputs/detail.txt` và upload làm artifact.
- [x] Bonus 4 - Hoàn trả về phiên bản trước: So sánh F1 model mới với report trên S3 và dừng release nếu F1 giảm.
- [x] Bonus 5 - Cảnh báo lệch lạc dữ liệu: So sánh tỷ lệ target dương với mức tham chiếu 24,8% và ghi cảnh báo vào report.
