# Các cải tiến của src_python_gpu_SA_RCRS_GRASP so với src_python

Các cải tiến của `src_python_gpu_SA_RCRS_GRASP` so với mã nguồn gốc `src_python` (base) được chia thành hai nhóm chính: **Cải tiến thuật toán** và **Cải tiến thực thi GPU**.

---

## 1. Cải tiến thuật toán

### 1.1. Khởi tạo toàn diện SA_RCRS_GRASP kết hợp Deep Local Search
Quá trình khởi tạo mỗi cá thể trong quần thể được thực hiện hoàn toàn trên GPU theo chuỗi 5 bước:
1. **RCRS-GRASP construction**: Kết hợp heuristic RCRS với danh sách ứng viên giới hạn (RCL) ngẫu nhiên hóa theo tham số $\alpha \in [\alpha_{lo}, \alpha_{hi}]$ để tạo các nghiệm ban đầu đa dạng.
2. **Feasibility check & Repair**: Tự động phát hiện và khắc phục vi phạm tải trọng, cửa sổ thời gian hoặc khách hàng bị thiếu/trùng lặp.
3. **Simulated Annealing Warmup**: Tinh chỉnh nghiệm ban đầu qua các toán tử nhiễu loạn lân cận với cơ chế làm nguội SA.
4. **Route Elimination**: Loại bỏ các tuyến ngắn (ít khách) và chèn lại khách vào các tuyến khác để giảm số lượng xe (NV) ngay từ đầu.
5. **Deep Local Search**: Áp dụng tìm kiếm cục bộ sâu (2 passes) để tối ưu hóa quãng đường (TD) trước khi bước vào thế hệ 1.

### 1.2. Phân cấp Lexicographic trong Cơ chế chấp nhận SA & Best Selection
So với hàm mục tiêu scalar $Cost = 2000 \cdot NV + TD$ của base:
- **Phân cấp mục tiêu**: Số xe (NV) là mục tiêu tối thượng bậc 1, Tổng quãng đường (TD) là mục tiêu bậc 2.
- **Quy tắc chấp nhận nghiệm**:
  - Nghiệm làm tăng số xe ($NV_{new} > NV_{old}$): **Tuyệt đối bị từ chối**.
  - Nghiệm làm giảm số xe ($NV_{new} < NV_{old}$): **Chấp nhận 100%**.
  - Khi cùng số xe ($NV_{new} == NV_{old}$):
    - Chấp nhận ngay nếu $TD_{new} \le TD_{old} + 10^{-3}$.
    - Nếu $TD_{new} > TD_{old}$, xác suất chấp nhận theo phân bố Metropolis:
      $$P = \exp\left(-\frac{TD_{new} - TD_{old}}{10^{-6} + T \cdot TD_{old}}\right) \quad \text{với } T = 1 - \frac{t}{t_{max}}$$
- **Ý nghĩa**: Khắc phục triệt để hiện tượng "loãng nhiệt" khi chia cho $Cost \approx 21000$ (khiến xác suất chấp nhận nghiệm xấu $> 99\%$), giúp thuật toán hội tụ mạnh mẽ và ép sâu giá trị TD ở các thế hệ sau.

### 1.3. Tập toán tử láng giềng sâu (Deep Local Search) kết hợp Inter-route Relocate
Bên cạnh các toán tử nội tuyến và hoán đổi cơ bản, GPU hỗ trợ đầy đủ 5 toán tử láng giềng:
1. **Intra-route 2-opt**: Đảo ngược đoạn đường con trong cùng một tuyến.
2. **Intra-route Relocate**: Dời một khách hàng sang vị trí khác trong cùng tuyến.
3. **Inter-route Swap**: Hoán đổi vị trí của hai khách hàng thuộc hai tuyến khác nhau.
4. **Inter-route 2-opt\***: Tráo đổi hai phần đuôi (tails) giữa hai tuyến xe.
5. **Inter-route Relocate**: Rút một khách hàng từ tuyến $r_1$ và chèn vào vị trí có lợi nhất trên tuyến $r_2$ (thỏa mãn tải trọng và thời gian). Toán tử này hỗ trợ đắc lực cho việc làm rỗng tuyến để giảm NV và rút ngắn TD.

### 1.4. Phủ sóng Local Search & Route Elimination theo đảo (Island Scope) hoặc toàn quần thể
- Cứ mỗi chu kỳ (mặc định 25 thế hệ), GPU thực thi `route_elimination` và `local_search`.
- Hỗ trợ chế độ `island` (tinh chỉnh 4 cá thể tinh hoa đại diện 4 đảo, cân bằng tối ưu giữa thời gian chạy ~120s/run và chất lượng nghiệm) hoặc `population` (toàn bộ 32 cá thể).
- Giúp các đảo luôn duy trì chất lượng nghiệm cao, cung cấp nguồn cha mẹ ưu tú và đa dạng cho các bước lai ghép tiếp theo.

### 1.5. Toán tử lai ghép SHO Guided Crossover dùng Greedy Best-Insertion
- Kế thừa 1–2 tuyến tinh hoa từ cá thể tốt nhất của đảo (`ibest`).
- Với các khách hàng chưa phục vụ từ nghiệm hiện tại và bạn phối ngẫu (peer): Áp dụng **Greedy Best-Insertion** với bộ lọc cắt tỉa khoảng cách tam giác ($\Delta d_{approx} < \text{best\_}\Delta d$) trước khi kiểm tra ràng buộc thời gian/tải trọng, giúp tốc độ chèn tăng hơn 10 lần.
- Chỉ mở tuyến mới khi không thể chèn vào bất kỳ tuyến hiện có nào, tránh tạo ra các tuyến zíc-zắc có TD cao như cơ chế chèn tuần tự (sequential packing).
- Cơ chế từ chối nghiệm không hợp lệ $O(1)$: Nếu cá thể con sau lai ghép không khả thi, thuật toán từ chối ngay lập tức thay vì chạy vòng lặp sửa nghiệm nặng nề ($O(N^3)$ repair), loại bỏ hoàn toàn phân kỳ luồng (warp divergence) trên GPU.

### 1.6. Toán tử khai thác WOA Intensification trực tiếp
- Khi $|A| < 1$ (giai đoạn bao vây thức ăn): Copy trực tiếp nghiệm tốt nhất của đảo (`ibest`) và áp dụng biến dị nhẹ 1–2 đỉnh để khai thác sâu xung quanh nghiệm ưu tú.
- Khi $|A| \ge 1$ (giai đoạn thăm dò): Copy nghiệm hiện tại và áp dụng đảo đoạn 2-opt ngẫu nhiên.
- Đánh giá khả thi trực tiếp và từ chối tức thời $O(1)$ nếu vi phạm, giữ nhịp độ thực thi đồng bộ trên GPU.

### 1.7. Mô hình đa đảo (Island Model) & Ring Migration
- Chia quần thể $P=32$ thành 4 đảo độc lập (mỗi đảo 8 cá thể).
- Định kỳ mỗi 20 thế hệ (`migration_interval`), nghiệm tốt nhất của đảo này được di cư sang đảo kế tiếp theo cấu trúc vòng tròn (Ring Topology) nếu tốt hơn cá thể kém nhất của đảo nhận.

### 1.8. Phá vỡ bế tắc theo từng đảo (Stagnation Diversification - Ruin & Recreate)
- Khi một đảo bị đình trệ (không cải thiện nghiệm tốt nhất), $40\%$ cá thể kém nhất trong đảo sẽ được phá vỡ (Ruin $20\%-40\%$ số khách hàng) và tái thiết lập (Recreate bằng Best-Insertion), bảo vệ cá thể tinh hoa không bị phá hủy.

---

## 2. Cải tiến thực thi GPU

### 2.1. Kiến trúc CUDA Graph 100% Resident (Zero Host Synchronization)
- Toàn bộ 1000 thế hệ tìm kiếm được đóng gói thành một CUDA Graph thực thi liên tục trên GPU.
- CPU chỉ chuẩn bị dữ liệu đầu vào (CPU_PREP) và giải mã nghiệm cuối cùng (CPU_DECODE), không đồng bộ hay can thiệp vào vòng lặp thế hệ.

### 2.2. Quản lý bộ nhớ Ping-pong và Scratch Buffers cố định
- Dùng 2 mảng quần thể `current` và `next` tráo đổi con trỏ (ping-pong swap), không cấp phát động hay giải phóng VRAM trong quá trình chạy.
- Toàn bộ dữ liệu scratch (tuyến tạm, cờ đánh dấu, danh sách chưa phục vụ) được định cỡ trước theo số khách $N$ và số cá thể $P=32$.

### 2.3. Trình tạo số ngẫu nhiên Xorshift128 độc lập trên GPU
- Mỗi luồng/cá thể sở hữu trạng thái RNG 128-bit riêng biệt trên device memory.
- Đảm bảo tính ngẫu nhiên chất lượng cao và khả năng tái lập hoàn toàn (reproducibility) qua từng run.

### 2.4. Theo dõi nghiệm tốt nhất phân tầng (Multi-tier Best Tracking)
- Cập nhật theo thứ bậc: `population` $\to$ `island_best` $\to$ `global_best` hoàn toàn trên GPU thông qua các kernel chuyên biệt với độ phức tạp cực thấp.

### 2.5. Kiểm soát số bước lặp và khử phân kỳ luồng (Warp Divergence Elimination)
- Trong `local_search`: Chặn cứng số lượt quét láng giềng (`max_passes = 2` trên GPU) và loại bỏ việc nhảy lại từ đầu (`continue`) mỗi khi tìm được một cải tiến nhỏ. Khi chạy test đối chiếu ngữ nghĩa CPU (`max_passes <= 0`), thuật toán tự động chuyển sang chế độ lặp đến khi hội tụ hoàn toàn.
- Khử hoàn toàn các lệnh gọi hàm sửa nghiệm `repair` lồng nhau trong thân vòng lặp tiến hóa; chỉ giữ `repair` ở bước khởi tạo quần thể ban đầu.
- Nhờ đó, thời gian 1 run trên instance 100 khách (`cdp103`, `rcdp101`) duy trì ở mức ~120s (thay vì bị treo hoặc chạy >500s do luồng warp bị stall).

### 2.6. Hỗ trợ CPU Reference tương đương 1:1
- Toàn bộ logic kernel và toán tử đều hỗ trợ thực thi trên Numba CPU Reference, giúp kiểm thử, gỡ lỗi và đối chiếu chính xác ngay cả trên các môi trường không có GPU phần cứng.