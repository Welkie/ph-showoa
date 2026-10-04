# Tổng hợp Các Cải tiến GPU trong `src_python_gpu_SA_RCRS_GRASP` so với Bài báo & Code Baseline (`PH-SHOWOA`)

Tài liệu này đối chiếu trực tiếp, khách quan và chính xác giữa phiên bản GPU (`src_python_gpu_SA_RCRS_GRASP`) với bài báo gốc **PH-SHOWOA** (*PLOS ONE 2026, doi: 10.1371/journal.pone.0343262*) và mã nguồn baseline (`src_python_baseline`).

Dưới đây là các cải tiến kỹ thuật cốt lõi hiện có của phiên bản GPU kèm phân tích nguyên lý giải thích **tại sao cải tiến GPU lại vượt trội hơn**:

---

### 1. Chuyển đổi Toàn diện sang Kiến trúc 100% Full-GPU Resident (Thay thế CPU Baseline)
* **So với Baseline (Paper PH-SHOWOA & Mã nguồn CPU Baseline)**:
  * *Baseline*: Thuật toán chạy 100% trên CPU (`--backends cpu`). Vòng lặp 1000 thế hệ được thực thi bằng đa tiến trình CPU (`ProcessPoolExecutor`) hoặc đơn tiến trình. Quá trình tính toán bị giới hạn nặng nề bởi năng lực tính toán tuần tự của CPU và chi phí IPC giữa các tiến trình, khiến thời gian chạy rất lâu (mất hàng giờ cho mỗi instance).
  * *Cải tiến GPU*: Chuyển đổi toàn bộ giải thuật thành **100% Full-GPU Resident** trên CUDA. Toàn bộ 1000 thế hệ cư trú và tính toán khép kín trên VRAM của GPU theo chu trình:
    `CPU_PREP` (nạp dữ liệu bài toán lên GPU 1 lần duy nhất) $\to$ `RUN_GPU_BEGIN` $\to$ **1000 thế hệ thuần GPU (chạy liên tục trên VRAM, không có bất kỳ lệnh đồng bộ `cuda.synchronize()` hay kéo dữ liệu về Host nào giữa các thế hệ)** $\to$ `RUN_GPU_END` $\to$ `CPU_DECODE` (chỉ copy nghiệm tốt nhất và log telemetry đúng 1 lần khi kết thúc run).
* **Tại sao tốt hơn**:
  * Khai thác triệt để hàng nghìn nhân CUDA xử lý song song thay vì chỉ vài nhân CPU.
  * Tốc độ giải quyết 1000 thế hệ cho bài toán 100 khách hàng giảm từ hàng giờ trên CPU xuống chỉ còn khoảng ~120 giây trên card GPU phổ thông (Nvidia Tesla T4).

---

### 2. Thay thế Khởi tạo SA "Mù" bằng Quy trình SA_RCRS_GRASP kết hợp Ép Giảm Xe (Route Elimination)
* **So với Baseline (Paper PH-SHOWOA, Algorithm 2, Trang 11-12)**:
  * *Baseline*: Trong paper gốc, thuật toán khởi tạo là **Simulated Annealing thuần túy (Algorithm 2: The Initialization Procedure)**. Nó bắt đầu từ một nghiệm khởi tạo thô và chạy SA ngẫu nhiên rất lâu ($T_0=100.0, \alpha=0.95, T_{min}=0.1, itermax=100 \implies \approx 13.500$ phép thử ngẫu nhiên trên CPU). Quá trình này hoàn toàn là "thử sai ngẫu nhiên" (blind random perturbations), không có định hướng hình học không gian và hoàn toàn **không có cơ chế chủ động ép giảm số lượng xe**. Nghiệm ban đầu thường có số lượng xe lớn, làm lãng phí hàng trăm thế hệ tiến hóa sau đó chỉ để thu gọn xe.
  * *Cải tiến GPU*: Thay thế toàn bộ bằng quy trình khởi tạo 5 giai đoạn có định hướng chạy song song trên GPU:
    1. *RCRS-GRASP Construction*: Dựng các chùm tuyến hình quạt có định hướng rõ rệt thông qua danh sách ứng viên ngẫu nhiên RCL ($\alpha \in [\alpha_{lo}, \alpha_{hi}]$) kết hợp hàm phạt góc lệch hướng tâm ($rs\_pen$) và phạt đầy tải ($rc\_pen$).
    2. *Feasibility Check & Auto Repair*: Tự động kiểm tra và sửa lỗi vi phạm ràng buộc trên device.
    3. *SA Warmup Dịu nhẹ*: Sau khi đã có chùm tuyến đẹp từ GRASP, SA Warmup trên GPU chỉ đóng vai trò "rung lắc nhẹ" ($T_0=100.0, \text{cooling}=0.85, T_{min}=0.5, 25\text{ iters} \approx 800$ phép thử) để gọt dũa các nút giao cắt nhỏ mà **bảo toàn nguyên vẹn khung tuyến hình quạt** vừa dựng.
    4. *Route Elimination*: Thuật toán chuyên biệt quét tuần tự tìm và loại bỏ các tuyến ít khách, tái chèn khách sang các tuyến khác nhằm ép giảm số lượng xe ($NV$) ngay từ thế hệ 0 (Paper hoàn toàn không có bước này).
    5. *Deep Local Search*: Quét 2 lượt tối ưu hóa sâu để làm mịn quãng đường trước khi bước vào thế hệ 1.
* **Tại sao tốt hơn**:
  * Tạo ra các chùm tuyến hình quạt tối ưu ngay từ đầu thay vì trông cậy vào $13.500$ bước hoán đổi ngẫu nhiên kéo dài của baseline.
  * Ép số lượng xe $NV$ xuống mức tối thiểu ngay từ thế hệ 0 nhờ toán tử *Route Elimination*, giúp thuật toán tiết kiệm hàng trăm thế hệ tìm kiếm.

---

### 3. Đánh giá Nghiệm Phân cấp Lexicographic (Ưu tiên Tuyệt đối Số xe NV)
* **So với Baseline (Paper PH-SHOWOA, Algorithm 1 dòng 15-21, Algorithm 2 dòng 7-8, Mục 4.8 trang 18)**:
  * *Baseline*: Paper sử dụng hàm mục tiêu vô hướng gộp $\text{Cost} = 2000 \cdot NV + TD$. Khi tính xác suất chấp nhận Metropolis trong SA:
    $$P = \exp\left(-\frac{\Delta \text{Cost}}{10^{-6} + T \cdot |\text{Cost}|}\right)$$
    Do $|\text{Cost}| \approx 20.000 - 25.000$, mẫu số bị thổi phồng lên hàng nghìn đơn vị. Kết quả là khi nghiệm bị xấu đi (thậm chí tăng thêm 1 xe $\Delta \text{Cost} = +2000$), xác suất chấp nhận $P$ vẫn $> 80\% - 99\%$, khiến thuật toán nhận nghiệm xấu vô tội vạ và làm bùng nổ số lượng xe (NV inflation).
  * *Cải tiến GPU*: Phân cấp thứ tự từ vựng nghiêm ngặt: Số xe ($NV$) là ưu tiên tuyệt đối bậc 1, Quãng đường ($TD$) là ưu tiên bậc 2.
    * Nghiệm làm tăng số xe ($NV_{new} > NV_{old}$): **Bị từ chối 100%**.
    * Nghiệm làm giảm số xe ($NV_{new} < NV_{old}$): **Được chấp nhận 100%**.
    * Khi cùng số xe: Metropolis chỉ xét trên độ chênh lệch quãng đường $\Delta TD$ và chia cho chính $|TD_{old}| \approx 1000$ (thay vì 21000).
    2. Hiệu ứng "Lây nhiễm gen xấu" làm tuyệt chủng nghiệm tốt (Nguy hiểm nhất)
Trong thuật toán bầy đàn (SHO-WOA), các cá thể liên tục giao phối và lai ghép chéo với nhau:

Khi một cá thể tăng lên 11 xe, áp lực tải trọng của nó giảm hẳn (chia 100 khách cho 11 xe thì mỗi xe chở rất ít khách, thời gian cực kỳ thong thả so với xe chở khít khao của 10 xe).
Vì quá thong thả, cá thể 11 xe này rất dễ gọt dũa quãng đường ngắn xuống và có vẻ ngoài "rất đẹp".
Đến bước chọn cha mẹ để lai ghép (Crossover), cá thể 11 xe này được chọn và truyền cấu trúc 11 xe sang cho các con cháu.
Chỉ sau 20 - 30 thế hệ, cả 30 cá thể trong quần thể đều bị lây nhiễm thành 11 hoặc 12 xe!
Lúc này, bộ gen 10 xe ban đầu bị TUYỆT CHỦNG hoàn toàn. Trong toàn bộ quần thể không còn ai có cấu trúc 10 xe nữa, và thuật toán không còn cơ hội nào quay về 10 xe được nữa.
* **Tại sao tốt hơn**:
  * Loại trừ triệt để hiện tượng loãng nhiệt và lỗi nhận nghiệm tăng xe bừa bãi của baseline, luôn kiểm soát và ép chặt số lượng xe ở mức tối thiểu.

---

### 4. Mô hình Quần thể Đa Đảo (Island Model) & Di cư Vòng (Ring Migration)
* **So với Baseline (Paper PH-SHOWOA, Algorithm 1, Trang 10)**:
  * *Baseline*: Quần thể phẳng (single flat population). Mỗi chu kỳ lại inject nghiệm toàn cục (`publish_global_best`) vào toàn bộ quần thể, khiến các cá thể nhanh chóng giống hệt nhau và bị hút vào cùng một hố cực tiểu cục bộ (hội tụ sớm).
  * *Cải tiến GPU*: Quần thể được chia thành các Đảo độc lập (mỗi đảo 5 - 8 cá thể). Đấu loại và cập nhật nghiệm nội bộ đảo. Không broadcast global best vào các đảo mỗi thế hệ. Các đảo chỉ trao đổi gen định kỳ thông qua **Ring Migration** (đảo $i$ chuyển cá thể tốt nhất sang thay thế cá thể tệ nhất của đảo $i+1$).
* **Tại sao tốt hơn**:
  * Duy trì độ đa dạng di truyền lâu dài, chống hiện tượng sụp đổ gen sớm; khai thác tối đa tính song song của kiến trúc Warp/Block trên GPU.

---

### 5. Phá vỡ Đình trệ Cục bộ Độc lập theo Từng Đảo (Island Stagnation Ruin & Recreate)
* **So với Baseline (Paper PH-SHOWOA, Algorithm 1 dòng 44-47, Trang 10)**:
  * *Baseline*: Khi toàn bộ tiến trình không cải thiện sau 50 thế hệ (`noImprove >= 50`), baseline gọi `DIVERSIFY(population)` để phá vỡ 40% cá thể ngẫu nhiên trên **toàn bộ quần thể phẳng**.
  * *Cải tiến GPU*: Theo dõi số thế hệ đình trệ độc lập cho từng đảo. Nếu đảo nào kẹt 50 thế hệ không cải thiện `island_best`, chỉ tiến hành Ruin (xóa 20% - 40% khách) và Recreate (chèn lại tối ưu) cho 40% cá thể yếu nhất của riêng đảo đó, giữ nguyên cá thể tinh hoa và các đảo khác đang phát triển tốt.
* **Tại sao tốt hơn**:
  * Tránh phá hỏng nghiệm của các đảo đang tiến hóa thuận lợi, dồn tài nguyên giải cứu có trọng điểm các đảo bị bế tắc.

---

### 6. Thực thi Song song Toàn Quần thể cho Bộ Toán tử Láng giềng trên GPU
* **So với Baseline (Paper PH-SHOWOA, Algorithm 8, Trang 18 & Trang 21)**:
  * *Baseline*: Paper có mô tả bộ Combined Local Search (gồm 2-opt, Relocate, Swap). Tuy nhiên, trang 10 (dòng 34-37) và trang 21 của paper khẳng định rõ ràng: do chạy trên CPU rất nặng, thuật toán **chỉ dám áp dụng Local Search định kỳ lên DUY NHẤT 1 cá thể tốt nhất toàn cục (`global_best`)** (*"restricts combined local search to periodic refinement of the global best solution"*). Toàn bộ các cá thể còn lại trong quần thể hoàn toàn không được tối ưu cục bộ. Ngoài ra, việc thử move trên CPU bằng cách clone đối tượng (`s.clone()`) rất tốn kém.
  * *Cải tiến GPU*: Đóng gói toàn bộ các toán tử láng giềng cốt lõi (Intra 2-opt, Intra Relocate, Inter Swap, Inter 2-opt*, Inter Relocate) thành các hàm CUDA Device Functions, chạy **song song đồng thời cho TOÀN BỘ quần thể ($P$ cá thể)** trên hàng trăm luồng GPU ở cả 3 giai đoạn: Khởi tạo (SA Warmup), Tiến hóa định kỳ (Periodic Local Search), và Phá vỡ đình trệ (Stagnation Diversification).
* **Tại sao tốt hơn**:
  * *Nâng cao chất lượng toàn quần thể*: Thay vì chỉ chăm chút 1 cá thể tinh hoa duy nhất, việc toàn bộ $P$ cá thể ở tất cả các đảo đều được tối ưu liên tục giúp kéo mặt bằng chất lượng nghiệm lên rất cao, cung cấp nguồn gen cha mẹ phong phú và ưu tú cho các phép lai ghép tiếp theo.
  * *Tốc độ vượt trội (Zero Object Allocation)*: Mọi phép thử đảo đoạn, hoán đổi khách hay dời tuyến đều thao tác trực tiếp trên mảng VRAM 3D `nodes[P, R, L]` và tính toán độ chênh lệch quãng đường $\Delta d$ trong $O(1)$ qua ma trận `dist_matrix`. Triệt tiêu hoàn toàn chi phí `clone()` đối tượng và bộ gom rác Garbage Collector của Python CPU.

---

### 7. Quản lý Bộ nhớ Ping-Pong và RNG Device Độc lập
* **So với Baseline (Paper PH-SHOWOA)**:
  * *Baseline*: Cấp phát và hủy đối tượng `Solution`, `Route` liên tục trên heap của Python, gây áp lực lớn cho bộ gom rác Garbage Collector (GC) làm giật lag thời gian chạy.
  * *Cải tiến GPU*: Cấp phát tĩnh trước 2 mảng quần thể `current` và `next`, hoán đổi con trỏ qua cơ chế ping-pong swap (zero-copy). Mỗi luồng GPU sở hữu trạng thái sinh số ngẫu nhiên Xorshift128 riêng trên VRAM.
* **Tại sao tốt hơn**:
  * Triệt tiêu 100% overhead của Garbage Collector, bộ nhớ VRAM ổn định tuyệt đối; đảm bảo tính tái lập nghiệm (reproducibility) 100% khi cố định seed.


