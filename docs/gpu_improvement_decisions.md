# Cải tiến bên trong flow SA_RCRS_GRASP trên GPU

> Sau khi người dùng yêu cầu triển khai và cung cấp bảng P36, một phần đề xuất bên dưới đã được áp dụng. Trạng thái code và kiểm chứng mới nhất nằm trong `gpu_p32_changes.md`; phần còn lại của tài liệu này là quyết định ở thời điểm phân tích.

## Phạm vi đã chốt

- `src_python` là nền tảng logic và thông số, đối chiếu nhánh paper_flags.
- Flow chính: base → khởi tạo SA_RCRS_GRASP → thực thi GPU.
- Population luôn 32.
- Cho phép cải tiến nhỏ bên trong flow. Mọi thay đổi khác base phải được mô tả; phân biệt cải tiến có bằng chứng, giả thuyết và sửa lỗi.
- `d88da6eb892e3ccc3f17d4fdcb024104d0ef516e` là nguồn tham khảo cải tiến, không phải chuẩn để rollback toàn bộ.
- Không yêu cầu tạo thêm một chế độ GPU baseline thuần trước khi cải thiện flow chính.

## Quyết định kỹ thuật đề xuất

| Phần | Khác base như thế nào | Đề xuất | Mức bằng chứng |
|---|---|---|---|
| Cost 2000*NV+TD | Không khác | Giữ nguyên trong tất cả operator/acceptance/best tracking | Source base |
| SA, lịch SHO/WOA | Công thức hiện chủ yếu theo base | Giữ công thức và thông số; sửa epsilon acceptance bị lệch | Có phản ví dụ base/GPU cho epsilon |
| RCRS-GRASP initialization | Thêm RCL ngẫu nhiên vào construction | Giữ làm cải tiến chính; mô tả rõ SA giữ theo base | Chưa có A/B chất lượng đầy đủ |
| Inter-route 2-opt* | Thêm đổi đuôi giữa hai tuyến vào local search | Ưu tiên khôi phục như neighborhood bổ sung, giữ feasibility và scalar cost | Probe 100 case, xem dưới |
| Route elimination | Thêm bước xóa tuyến và chèn lại khách | Giữ có điều kiện giảm scalar cost, không ưu tiên NV bất chấp cost | Đã có code và test chặn tăng scalar cost; chưa đo lợi ích tổng thể |
| Local search trên best mỗi island | Thêm nhiều nghiệm xuất phát thay vì chỉ run-best | Ưu tiên thử trước full-population nếu dùng islands | Giả thuyết; chưa benchmark |
| Local search toàn population | Bản cũ cải thiện từng cá thể, kể cả sau init | Là ứng viên mạnh hơn nhưng tốn thời gian; chỉ chọn sau A/B | Chưa benchmark tương đương ngân sách |
| Islands/ring migration | Peer, guide và elite chuyển từ toàn quần thể sang mỗi island | Giữ là cải tiến tùy cấu hình; với P32 thử rõ I4 (8 cá thể/island) so với I1 | Chưa chứng minh I4 tốt hơn I1 |
| RCRS capacity proxy | Thay tc và hệ số chuẩn hóa của base | Không mặc định giữ vì nhanh; so với criterion base dưới cùng GRASP | Chưa chứng minh tốt hơn |
| Uniform coefficients | Thay Latin sampling của base | Ưu tiên sampling base; uniform là biến thể cần A/B riêng | Chưa chứng minh tốt hơn |
| Diversification | Base: stagnation-triggered; cũ: periodic; islands đổi phạm vi elite | Giữ trigger và tỷ lệ base; phạm vi theo island được khai báo cùng island model | Cơ chế cũ có lỗi metadata, không sao chép nguyên |
| CUDA Graph/device buffers | Thay cơ chế thực thi | Giữ; đưa operator cải tiến vào graph, không cần quay về host generation loop | Simulator kiểm tra schedule, chưa đo GPU thật |

“Giữ/khôi phục/thử” là kết luận phân tích, không có nghĩa solver đã được sửa trong lần rà này.

## Kiểm chứng bổ sung: local search cũ trên điểm dừng của local search hiện tại

Chạy `scripts/probe_td_neighborhood.py --batch --cases 100` bằng Python trong .venv.

- 100 bài toán sinh từ seed 20260923, mỗi bài 16 khách, capacity 4, delivery 1, pickup 0, time windows rộng, 4 xe.
- Với mỗi bài: chạy local search hiện tại đến điểm dừng, rồi chạy local search cũ 2 passes trên chính nghiệm đó.
- Đánh giá lại nghiệm bằng evaluator hiện tại, bao gồm coverage và feasibility.
- 100/100 nghiệm hợp lệ; 13/100 giảm TD; 0/100 tăng TD.
- Mean TD trước: 634.6533861123; sau: 631.1809588773. Mức giảm của hai trung bình khoảng 0.547%.
- Trường hợp case 5 đã kiểm chứng riêng có phép đổi đuôi trực tiếp giảm TD tại điểm dừng của local search hiện tại. Sau full old local search, TD giảm từ 772.7120002171 xuống 743.5255347494, cùng NV=4.

Kết quả chi tiết: `td_neighborhood_probe.json`.

Giới hạn: đây là thử bổ sung old local search sau current local search, không phải so hai solver end-to-end hoặc A/B chỉ riêng 2-opt*. 100 case là các bài toán sinh đơn giản, không phải 100 seeds chạy solver với P32. Probe dùng một nghiệm/bài, bỏ JIT để kiểm tra logic; không đo tốc độ. Muốn kết luận hiệu quả của 2-opt* riêng cần gắn nó vào current local search và chạy ablation. Không suy diễn từ 13/100 rằng dataset VRPSPDTW thực tế sẽ có tỷ lệ tương tự.

## Các sai lệch phải tách khỏi danh sách cải tiến

1. GPU SA thiếu điều kiện base `deltaCost <= PRECISION`: sửa code và test oracle theo base.
2. Notebook override SA iterations=25 so với base=100: dùng 100 cho cấu hình theo base, trừ khi công bố đó là tuning riêng.
3. P32/I6 âm thầm fallback I1: cấu hình phải ghi rõ I1 hoặc I4; không đổi population lên 36. Một island tự nó không phải lỗi.
4. Bản cũ bỏ kết quả trả về eval_solution sau diversification, làm metadata có thể không khớp tuyến: không phục hồi lỗi này.
5. `init` trong cấu hình không thực sự chọn nhánh kernel hiện tại: log và validation phải phản ánh flow SA_RCRS_GRASP đang chạy.

## Thứ tự đánh giá trong chính flow GPU

1. Cố định P32; công bố rõ islands, thông số SA, seed list, instance, max_iter và runs. Sửa các sai lệch base có bằng chứng.
2. Thêm 2-opt* vào cuối chuỗi neighborhood hiện tại, chỉ commit move hợp lệ và giảm scalar cost; khi cải thiện thì restart như base. Không thay neighborhood có sẵn.
3. Đánh giá islands I1/I4 và local-search scope theo từng thí nghiệm, không thay cả hai rồi gán lợi ích cho một yếu tố.
4. Đánh giá criterion chính xác của base so với proxy, rồi Latin sampling so với uniform; giữ GRASP trong cả hai vế.
5. Chỉ chốt cấu hình sau khi có NV/TD/cost tính lại từ tuyến, feasibility, thời gian và thống kê nhiều runs. So cả cùng số thế hệ lẫn ngân sách thời gian.

Chưa thể cam kết TD bằng hoặc tốt hơn d88da6e khi chưa có benchmark thực tế cùng cấu hình và các tuyến cũ để xác minh. Kết quả hiện đủ để ưu tiên 2-opt* và sửa lệch base; chưa đủ để chốt islands, proxy hoặc phạm vi local search tối ưu.
