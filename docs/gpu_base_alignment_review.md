# Rà soát GPU theo src_python, population 32

> Phạm vi cuối cùng đã được người dùng xác nhận: flow chính là base + SA_RCRS_GRASP + GPU, vẫn cho phép các cải tiến nhỏ có mô tả và kiểm chứng. Xem `gpu_improvement_decisions.md`. Các đề xuất baseline thuần bên dưới chỉ phục vụ đối chiếu, không phải yêu cầu tạo thêm sản phẩm hoặc loại bỏ mọi extension hiện có.

## Chuẩn đối chiếu

Theo yêu cầu người dùng, `src_python` hiện tại là code base chuẩn. Không dùng `d88da6e` làm chuẩn thuật toán; chỉ khai thác các ý tưởng của bản cũ dưới dạng cải tiến được khai báo. Population cố định 32.

Base có hai nhánh `paper_flags`; GPU ép `paper_flags=True` và các kernel được viết theo nhánh đó. Vì vậy đối chiếu trong tài liệu này là **src_python với --paper_flags --pop_size 32**. Không được tuyên bố GPU tương đương mọi chế độ của src_python. Đây là đối chiếu source local, không xác minh nguồn gốc lịch sử hay mức độ trùng với code xuất bản của tác giả.

## Những phần thuộc base, phải giữ

| Thành phần | Hành vi chuẩn |
|---|---|
| Cost và best selection | 2000*NV + TD, so sánh scalar |
| Main-loop SA | deltaCost; T=1-t/max_iter; mẫu số epsilon+T*abs(currentCost); chấp nhận luôn delta<=PRECISION |
| Warmup SA paper mode | Score TD; T0=100, alpha=0.95, Tmin=0.1, itermax=100; lưu best-so-far |
| Warmup neighborhoods | Swap, insert, reverse, inter-relocate, inter-swap |
| Lịch hybrid | a=2-2*t/max_iter; pSHO=max(0.15,0.5*(1-t/max_iter)) |
| SHO | Tournament tối đa 3 peers khác cá thể hiện tại; crossover best/peer/current; mutation 0.35 |
| WOA | Encircling/random operation/spiral và relink theo nhánh paper mode |
| Local search | Run-best; interval 25, gen 1/26/51/...; first improvement, restart đến điểm dừng |
| Local-search neighborhoods | Intra 2-opt, relocate một khách, swap một khách giữa hai tuyến |
| Diversification | Sau 50 thế hệ không cải thiện; bảo vệ elite; chọn round(non_elite*0.4); ruin 20–40% khách rồi recreate |
| Feasibility | Giới hạn xe từ input, coverage, capacity, time windows; repair theo base |

Main-loop SA dùng cost nhưng warmup paper mode dùng TD: đây là khác biệt vốn có trong base, không tự ý đồng nhất hai score.

Việc bản cũ dùng lexicographic, cosine schedule, nhiệt khác hoặc local search toàn quần thể không phải lý do thay lại các hành vi trên. Bỏ 2-opt* là đúng tập neighborhood của `_combined_local_search` hiện tại; thêm nó là mở rộng thuật toán.

## Các điểm GPU hiện tại chưa thể gọi là bám hoàn toàn base

### 1. Ngưỡng chấp nhận SA lệch thật sự

Base `search_framework._sa_accept` chấp nhận vô điều kiện khi delta<=PRECISION (0.001). GPU `scalar_sa_probability` chỉ trả 1 ngay khi delta<0; phần còn lại dùng Boltzmann. `sho_woa_update_single` cũng chỉ chấp nhận ngay khi new_cost<old_cost.

Đã kiểm chứng trực tiếp: currentCost=4100, newCost=4100.0005, iteration=99, max_iter=100, RNG draw=0.99999999. Base chấp nhận; GPU probability=0.9999878049527039 nên từ chối. Đây là sai lệch cần sửa theo base, không phải cải tiến. Test SA hiện tại chủ động yêu cầu không có epsilon acceptance nên cũng phải sửa oracle theo base. Sai lệch nhỏ này chưa được chứng minh gây chênh TD lớn.

### 2. Notebook đang ghi đè tham số SA

Notebook đặt `--sa_iterations 25`. Engine lấy tham số này thay cho sa_itermax; base mặc định 100. Nếu yêu cầu giữ thông số base thì cấu hình benchmark GPU phải dùng 100, hoặc bỏ override để nhận mặc định đồng nhất. T0/alpha/Tmin hiện đã cùng base.

### 3. Construction không chỉ thêm GRASP

Base `operator.criterion`: deltaTD + lambda*tc*(2*max_dist-min_dist) - gamma*rs. tc dùng residual capacity theo vị trí chèn và tỷ lệ nhu cầu chưa phục vụ.

GPU: deltaTD + lambda*rc_pen - gamma*rs, với rc_pen=max(0,max_load+max(delivery,pickup)-0.7*capacity). Đồng thời lấy lambda/gamma độc lập từ uniform RNG thay vì dùng `data.latin` đã tạo ở Data. Base với init=sa dùng Latin grid đã shuffle/cắt đủ p_size (32 vẫn được hỗ trợ).

Đây là ba thay đổi độc lập: GRASP/RCL, proxy capacity, và sampling. Không được gộp cả ba dưới nhãn “RCRS giống base”. Khuyến nghị dùng chính criterion và sampling của base làm nền, rồi chỉ thêm GRASP như cải tiến riêng. Proxy có thể là biến thể tối ưu thời gian nhưng chưa có bằng chứng giúp TD.

### 4. Init selector chưa tạo được chế độ baseline thuần

Kernel initialization luôn chạy RCRS-GRASP → repair → SA → route elimination, không phân nhánh theo `data.init`. `--init sa` hay `--init rcrs` không đủ để phục hồi initialization của base. Route elimination cũng luôn chạy trước periodic local search. Muốn có baseline GPU tương đương base cần công tắc thực sự cho các phần mở rộng, không chỉ đổi tên init trong log.

### 5. Islands thay đổi phạm vi các operator

Base chọn peer trên toàn bộ quần thể và dùng run-best làm guide. GPU dùng peer trong island và island-best; diversification bảo vệ elite theo island. Đây là thay đổi thuật toán cần khai báo cùng island model, không chỉ là cách chia việc lên GPU.

Với P=32 và default islands=6, engine âm thầm chuyển về 1 island. Một island gần cấu trúc quần thể base hơn, không phải bản thân nó là lỗi thuật toán. Điều chưa tốt là cấu hình yêu cầu và cấu hình thực thi khác nhau. Nên ghi rõ islands=1 cho baseline; nếu bật cải tiến có thể thử islands=4, mỗi island 8 cá thể. Không đổi P lên 36. Chưa có benchmark chứng minh 4 islands tốt hơn 1 hoặc 8.

### 6. Khác biệt thực thi và trường hợp biên

GPU dùng Xorshift128 và seed mỗi run=seed+run*100003; base dùng random.Random, run_seed=seed+run-1 và seed theo worker. Đây là khác biệt triển khai cần công bố: cùng seed CLI không đảm bảo cùng chuỗi nghiệm. Tests cùng random stream dùng để kiểm tra operator; nhiều seeds và ngân sách tính toán dùng để so chất lượng.

Repair có cách xử lý an toàn khác ở một số trường hợp biên (ví dụ return-to-depot time violation và rollback sau mutation thất bại); các tests parity nhỏ hiện có không đủ chứng minh tương đương mọi đầu vào. Nên ghi rõ behavior được chấp nhận thay vì gọi chung toàn bộ repair là đã khớp tuyệt đối.

## Danh mục cải tiến phải tách rõ

| Phần mở rộng | Trạng thái | Cách diễn giải |
|---|---|---|
| GRASP/RCL trên RCRS | Đã có | Cải tiến đa dạng hóa initialization; nên giữ criterion base ở dưới |
| Capacity proxy thay tc | Đã có | Xấp xỉ thuật toán, chưa chứng minh tăng chất lượng; không mặc nhiên coi tốt hơn |
| Uniform lambda/gamma thay Latin sampling | Đã có | Biến thể sampling, cần đánh giá độc lập |
| Route elimination | Đã có | Bước search thêm, chỉ giữ khi scalar cost giảm |
| Islands + ring migration + island-scoped selection/diversification | Đã có, nhiều islands không hoạt động với default P32/I6 | Một nhóm mở rộng thuật toán; cấu hình P32/I4 là ứng viên |
| Inter-route 2-opt* | Chưa có ở current local search | Ứng viên tốt: witness từ lần rà trước giảm TD 3.78%, cùng NV; chưa chứng minh trên benchmark |
| Local search best mỗi island hoặc toàn quần thể | Chưa có trong periodic schedule hiện tại | Ứng viên mở rộng, tăng chi phí; không thay thế baseline run-best mặc định |
| CUDA Graph, device buffers, ping-pong, device scheduling, device RNG | Đã có | Cải tiến triển khai/hiệu năng; Graph/buffers không tự tạo thuật toán tối ưu TD mới |

## Cấu hình đối chiếu đề xuất

Giữ P=32 ở tất cả biến thể, scalar objective, SA 100/0.95/0.1/100, LS interval 25, stagnation 50, mutation .35, diversify .4. Dùng cùng max_iter/runs; nếu chọn paper defaults thì 1000 iterations và 30 runs, còn population override rõ là 32.

1. Base CPU: paper_flags, P32, init SA của base.
2. GPU baseline: cùng logic, P32, một quần thể chung; chưa tồn tại chế độ thuần đầy đủ trong code hiện tại.
3. GPU + GRASP (criterion/sampling giữ base).
4. Thêm route elimination, rồi 2-opt*, rồi islands=4; đo từng phần độc lập hoặc ablation rõ ràng.

Không đổi cost hay nhiệt SA để chạy đua với d88da6e. Chất lượng của cải tiến phải được đánh giá trên nền base cố định, ghi cả NV/TD/cost/feasibility/thời gian và nhiều seeds. GPU nhanh hơn và nghiệm tốt hơn là hai kết luận cần đo riêng.

Lần rà này chỉ cập nhật phân tích, không sửa solver/notebook và không chỉnh `docs/improved.md` đang có thay đổi của người dùng.
