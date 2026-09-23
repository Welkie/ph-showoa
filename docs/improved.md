# So sánh Chi tiết: `src_python_gpu_SA_RCRS_GRASP` vs `src_python_baseline`

Tài liệu này đối chiếu toàn diện giữa mã nguồn hiện tại (`src_python_gpu_SA_RCRS_GRASP`) và mã nguồn gốc (`src_python_baseline`). Văn bản được chia làm 2 phần rõ rệt:
1. **Phần 1: Các điểm cải tiến nổi bật** (Nâng cấp thuật toán & Tối ưu hóa GPU mang lại đột phá về chất lượng nghiệm và tốc độ).
2. **Phần 2: Bảng đối chiếu thông số & Các điểm bị lệch / chưa sát với baseline** (Phân tích chi tiết từng hằng số, ngưỡng dung sai, logic thuật toán và đánh giá: *cái nào là cải tiến cần giữ*, *cái nào chưa fit cần code lại để chuẩn hóa theo baseline*).

---

# PHẦN 1: CÁC ĐIỂM CẢI TIẾN TRONG `src_python_gpu_SA_RCRS_GRASP`

### 1.1. Kiến trúc Thực thi 100% Full-GPU Resident (Zero Host Synchronization)
- **Baseline**: 
  - Vòng lặp tiến hóa chạy trên CPU; việc tính toán đánh giá nghiệm hoặc chạy song song sử dụng `ProcessPoolExecutor` (đa tiến trình CPU) hoặc gọi GPU làm proxy trung gian tính điểm từng batch.
  - Xảy ra nghẽn cổ chai liên tục do chi phí giao tiếp IPC giữa các tiến trình CPU và độ trễ copy dữ liệu Host-to-Device (H2D) / Device-to-Host (D2H).
- **Cải tiến GPU**:
  - Toàn bộ 1000 thế hệ tìm kiếm được đóng gói hoàn toàn trong các CUDA Kernels chạy liên tục trên GPU.
  - Chu trình dữ liệu theo đúng chuẩn: **`CPU_PREP`** (cấp phát VRAM, nạp dữ liệu bài toán 1 lần duy nhất) $\to$ **`RUN_GPU_BEGIN`** $\to$ **Vòng lặp GPU cư trú 100% trên VRAM** (không có bất kỳ lệnh đồng bộ `cuda.synchronize()` hay `copy_to_host()` nào giữa các thế hệ) $\to$ **`RUN_GPU_END`** $\to$ **`CPU_DECODE`** (chỉ copy nghiệm tốt nhất và log telemetry về Host đúng 1 lần khi kết thúc run).
  - Tốc độ tăng tốc vượt trội: giải quyết 1000 thế hệ trên instance 100 khách hàng chỉ trong ~120s trên card GPU phổ thông (Nvidia Tesla T4).

### 1.2. Khởi tạo Toàn diện SA_RCRS_GRASP kết hợp Route Elimination & Deep Local Search
- **Baseline**:
  - Khởi tạo đơn giản bằng RCRS cổ điển hoặc SA cơ bản trên CPU.
  - Nghiệm ban đầu thường có số lượng xe (NV) lớn và quãng đường (TD) chưa được tối ưu, khiến thuật toán mất hàng trăm thế hệ đầu chỉ để thu gọn nghiệm.
- **Cải tiến GPU**:
  - Chuỗi 5 giai đoạn khởi tạo chạy song song trên $P$ luồng GPU:
    1. **RCRS-GRASP Construction**: Kết hợp điểm đánh giá RCRS với danh sách ứng viên ngẫu nhiên RCL theo ngưỡng $\alpha \in [\alpha_{lo}, \alpha_{hi}]$ để đa dạng hóa nghiệm.
       $$\text{Score} = \Delta TD + 0.5 \cdot rc\_pen + 0.3 \cdot rs\_pen$$
       *(phạt các tuyến gần đầy tải $rc\_pen$ và phạt độ lệch hướng tâm $rs\_pen$ để tạo các chùm tuyến hình quạt gọn gàng)*.
    2. **Feasibility Check & Auto Repair**: Kiểm tra và sửa lỗi vi phạm tải trọng, cửa sổ thời gian trực tiếp trên GPU.
    3. **Simulated Annealing Warmup**: Nhảy thoát cục bộ với cơ chế làm nguội SA qua 5 toán tử láng giềng.
    4. **Route Elimination**: Quét tuần tự loại bỏ các tuyến ít khách, tái chèn sang các tuyến khác nhằm ép giảm số lượng xe (NV) ngay từ thế hệ 0.
    5. **Deep Local Search**: Gọt dũa quãng đường (TD) qua 2 lượt quét có hệ thống trước khi bước vào tiến hóa.

### 1.3. Phân cấp Mục tiêu Lexicographic (NV-First, TD-Second)
- **Baseline**:
  - Sử dụng hàm mục tiêu vô hướng gộp: $\text{Cost} = 2000.0 \cdot NV + TD$.
  - Khi tính xác suất chấp nhận Metropolis trong Simulated Annealing:
    $$P = \exp\left(-\frac{\Delta \text{Cost}}{10^{-6} + T \cdot |\text{Cost}|}\right)$$
    Do $|\text{Cost}| \approx 20000 - 25000$, mẫu số bị thổi phồng lên hàng nghìn đơn vị. Kết quả là khi nghiệm bị xấu đi (ví dụ $\Delta TD = +50$ hoặc thậm chí tăng thêm 1 xe $\Delta \text{Cost} = +2000$), xác suất chấp nhận $P$ vẫn $> 80\% - 99\%$, khiến thuật toán nhận nghiệm xấu vô tội vạ và làm bung số lượng xe (NV inflation).
- **Cải tiến GPU**:
  - Phân cấp thứ tự từ vựng rõ ràng: Số xe ($NV$) là ưu tiên tuyệt đối bậc 1, Quãng đường ($TD$) là ưu tiên bậc 2.
  - **Quy tắc chấp nhận nghiệm Lexicographic**:
    - Nghiệm làm tăng số xe ($NV_{new} > NV_{old}$): **Bị từ chối 100%**.
    - Nghiệm làm giảm số xe ($NV_{new} < NV_{old}$): **Được chấp nhận 100%**.
    - Khi cùng số xe ($NV_{new} == NV_{old}$):
      - Nếu $TD_{new} \le TD_{old} + 10^{-3}$: Chấp nhận ngay lập tức.
      - Nếu $TD_{new} > TD_{old}$: Xét Metropolis trên độ chênh lệch quãng đường $\Delta TD$ và chia cho chính $|TD_{old}|$ ($\approx 1000$ thay vì 21000):
        $$P = \exp\left(-\frac{\Delta TD}{10^{-6} + T \cdot |TD_{old}|}\right)$$
  - Giúp thuật toán kiểm soát chặt chẽ số xe và hội tụ sâu về quãng đường tối ưu.

### 1.4. Đấu loại 3 Cá thể theo Thứ tự Từ vựng (`is_better_lex`) trong Chọn Cha Mẹ
- **Baseline**: Đấu loại ngẫu nhiên $k=3$ cá thể dựa trên giá trị `pop_fit` vô hướng.
- **Cải tiến GPU**: 
  - Chọn 3 cá thể ngẫu nhiên và so sánh bằng hàm `is_better_lex(nr, dist)`. 
  - Luôn chọn cá thể có số xe ít hơn; nếu cùng số xe mới xét đến quãng đường ngắn hơn. Đảm bảo nguồn gen của cha mẹ ưu tú luôn được di truyền qua các thế hệ.

### 1.5. Bộ Toán tử Láng giềng Sâu Đầy đủ (5 Operators) trên GPU
- **Baseline**: Chủ yếu dùng 2-opt cơ bản và 2-opt\*, thiếu các toán tử liên tuyến tinh tế.
- **Cải tiến GPU**: Hỗ trợ đầy đủ 5 toán tử láng giềng chạy song song trên device:
  1. *Intra-route 2-opt*: Đảo ngược đoạn con trong cùng một tuyến.
  2. *Intra-route Relocate*: Dời một khách hàng sang vị trí khác trong tuyến.
  3. *Inter-route Swap*: Hoán đổi hai khách hàng giữa hai tuyến xe.
  4. *Inter-route 2-opt\**: Tráo đổi hai phần đuôi (tails) giữa hai tuyến xe.
  5. *Inter-route Relocate*: Rút một khách hàng từ tuyến này chèn sang vị trí tối ưu ở tuyến khác. Đây là toán tử cực mạnh giúp làm rỗng tuyến để giảm bớt xe.

### 1.6. Mô hình Quần thể Đa Đảo (Island Model) & Di cư Vòng (Ring Migration)
- **Baseline**: Quần thể phẳng (single flat population). Mỗi chu kỳ lại inject nghiệm toàn cục (`publish_global_best`) vào quần thể, rất dễ dẫn đến hội tụ sớm (premature convergence) khi cả quần thể bị hút vào cùng một cực tiểu cục bộ.
- **Cải tiến GPU**:
  - Quần thể được chia thành các Đảo độc lập (mỗi đảo 8 cá thể).
  - Không broadcast nghiệm toàn cục vào các đảo mỗi thế hệ để duy trì độ đa dạng gen tối đa.
  - Các đảo chỉ trao đổi thông tin định kỳ thông qua **Ring Migration** (di cư vòng): đảo $i$ chuyển cá thể tốt nhất sang thay thế cá thể kém nhất của đảo $(i+1) \pmod{\text{num\_islands}}$.

### 1.7. Phá vỡ Bế tắc Cục bộ theo Từng Đảo (Island Stagnation Ruin & Recreate)
- **Baseline**: Khi toàn bộ quá trình tìm kiếm không cải thiện sau 50 thế hệ, phá vỡ 40% cá thể ngẫu nhiên trong toàn quần thể.
- **Cải tiến GPU**:
  - Theo dõi đình trệ độc lập cho từng đảo.
  - Nếu đảo nào bị kẹt 50 thế hệ không cải thiện `island_best`, tiến hành Ruin (xóa $20\%-40\%$ số khách) và Recreate (chèn lại tối ưu) cho 40% cá thể yếu nhất trong chính đảo đó, giữ nguyên cá thể tinh hoa và các đảo đang phát triển tốt khác.

### 1.8. Quản lý Bộ nhớ Ping-Pong và RNG Device Độc lập
- **Baseline**: Cấp phát và hủy đối tượng `Solution`, `Route` liên tục trên heap của Python, gây áp lực lớn lên Garbage Collector (GC).
- **Cải tiến GPU**: 
  - Cấp phát tĩnh trước 2 mảng quần thể `current` và `next`, tráo đổi con trỏ qua cơ chế ping-pong swap.
  - Mỗi luồng sở hữu trạng thái sinh số ngẫu nhiên Xorshift128 riêng trên GPU VRAM, đảm bảo khả năng tái lập nghiệm 100% (reproducibility) khi cố định seed.

---

# PHẦN 2: BẢNG ĐỐI CHIẾU THÔNG SỐ & CÁC ĐIỂM CHƯA SÁT VỚI BASELINE

Bảng dưới đây tổng hợp chi tiết tất cả các thông số, hằng số dung sai và nhánh logic giữa hai phiên bản:

| Hạng mục / Tham số | Baseline (`src_python_baseline`) | Hiện tại (`src_python_gpu_SA_RCRS_GRASP`) | Bản chất khác biệt | Đánh giá & Kiến nghị |
| :--- | :--- | :--- | :--- | :--- |
| **Công thức $p_{hybrid}$** | $p_{hybrid} = \max(0.15, 0.5 \cdot (1 - \frac{t}{t_{max}}))$ (Tuyến tính, có sàn 0.15) | $p_{hybrid} = 0.5 \cdot (1 + \cos(\frac{\pi t}{t_{max}}))$ (Cosine decay, về 0.0) | **Chưa sát baseline** (Lệch công thức & thiếu sàn 0.15) | **Cần code lại để fit** (nếu muốn đúng chuẩn baseline) hoặc giữ có sàn `max(0.15, ...)` |
| **Dung sai so sánh (`PRECISION`)** | `PRECISION = 0.001` ($10^{-3}$) dùng xuyên suốt | Trong Local Search dùng $10^{-4}$ (`1e-4`), trong acceptance dùng $10^{-3}$ | **Chưa sát baseline** (Không nhất quán giữa $10^{-3}$ và $10^{-4}$) | **Cần code lại để fit** (Quy về thống nhất $10^{-3}$ theo `PRECISION` của baseline) |
| **Hàm mục tiêu chấp nhận SA** | Vô hướng: $\text{Cost} = 2000 \cdot NV + TD$, chia cho $|\text{Cost}| \approx 21000$ | Thứ tự từ vựng: Cấm tăng NV, Metropolis chỉ trên $\Delta TD$, chia cho $|TD| \approx 1000$ | **Cải tiến vượt bậc** (Loại trừ lỗi loãng nhiệt và tăng xe vô tội vạ) | **Khuyên GIỮ NGUYÊN** (Đây là cải tiến cốt lõi đem lại chất lượng nghiệm tốt) |
| **Tham số SA Warmup** | $T_0=100.0, \alpha=0.95, T_{min}=0.1$, `itermax=100` | $T_0=100.0, \text{cooling}=0.85, T_{min}=0.5$, `sa_iters=25` | **Chưa sát baseline** (Thu nhỏ để tăng tốc GPU) | **Xem xét**: Giữ nguyên cho GPU để tránh nghẽn thời gian khởi tạo; nếu nâng lên 100 iter sẽ tăng thời gian chạy |
| **Nới lỏng số xe (`max_num`)** | `self.vehicle.max_num = int(value)` (Nghiêm ngặt) | `int(value) + V_NUM_RELAX` (`V_NUM_RELAX = 3`) | **Chưa sát baseline** (Cho phép tạm thời dôi 3 xe khi dựng nghiệm) | **Khuyên GIỮ NGUYÊN**: Giúp giải phóng bế tắc khi chèn khách, miễn là nghiệm xuất ra luôn thỏa mãn $\le$ số xe đề bài |
| **Mở tuyến mới ngẫu nhiên khi chèn khách** | Có xác suất 15% (`rng.random() < 0.15`) mở tuyến mới nếu không bật `paper_flags` | Tuyệt đối không mở tuyến mới nếu còn chỗ chèn khả thi | **Cải tiến có chủ đích** (Tránh làm phình to số lượng xe NV) | **Khuyên GIỮ NGUYÊN**: Mở tuyến bừa bãi sẽ làm tăng NV, đi ngược lại mục tiêu giảm xe |
| **Cấu trúc Quần thể** | Quần thể phẳng $P=64$ (hoặc 30), đấu loại toàn quần thể, inject global best | Quần thể chia Đảo ($4 \times 8 = 32$), đấu loại nội bộ đảo, Ring Migration | **Cải tiến kiến trúc** (Chống hội tụ sớm, tăng tính đa dạng) | **Khuyên GIỮ NGUYÊN**: Cực kỳ phù hợp với mô hình song song khối luồng của GPU |
| **Tần suất in Log (`OUTPUT_PER_GENS`)** | `OUTPUT_PER_GENS = 1` (In mỗi thế hệ) | `OUTPUT_PER_GENS = 25` (In mỗi 25 thế hệ) | **Tối ưu hóa I/O** (Giảm nghẽn console) | **Khuyên GIỮ NGUYÊN**: In 1000 dòng log từ GPU về console sẽ làm chậm run |
| **Độ dài toán tử Or-Opt** | `DEFAULT_OR_OPT_LEN = 3` | `or_opt_len = 2` (khi bật cấu hình `paper_flags`) | **Chưa sát baseline** | **Cần code lại để fit** nếu muốn khớp tham số mặc định của baseline |

---

## CHI TIẾT CÁC ĐIỂM CẦN XEM XÉT VÀ CODE LẠI ĐỂ FIT VỚI BASELINE

Dưới đây là phân tích sâu về các điểm bạn có thể quyết định **code lại để fit hoàn toàn** với baseline:

### 1. Thông số tỷ lệ lai ghép $p_{hybrid}$ (Ngưỡng sàn 0.15)
- **Hiện trạng trong Baseline** (`src_python_baseline/search_framework.py:514`):
  ```python
  def _dynamic_parameters(iteration: int, max_iter: int) -> Tuple[float, float]:
      if max_iter <= 0:
          return 0.0, 0.15
      ratio = min(max(float(iteration) / float(max_iter), 0.0), 1.0)
      a = 2.0 - 2.0 * ratio
      p_hybrid = max(0.15, 0.5 * (1.0 - ratio))
      return a, p_hybrid
  ```
  - Tại thế hệ đầu ($ratio = 0$): $p_{hybrid} = 0.5$ (50% SHO, 50% WOA).
  - Tại thế hệ cuối ($ratio = 1$): $p_{hybrid} = \max(0.15, 0.0) = \mathbf{0.15}$ (**luôn giữ ít nhất 15% xác suất SHO**).
- **Hiện trạng trong GPU** (`src_python_gpu_SA_RCRS_GRASP/gpu_engine.py`):
  ```python
  p_hybrid = 0.5 * (1.0 + math.cos(math.pi * float(iter_idx) / float(max_iter)))
  ```
  - Tại thế hệ đầu: $p_{hybrid} = 1.0$ (100% SHO, 0% WOA).
  - Tại thế hệ cuối: $p_{hybrid} = 0.0$ (**tắt hoàn toàn SHO**, 100% WOA).
- **Đánh giá & Giải pháp**:
  - Việc tắt hoàn toàn SHO ở các thế hệ cuối khiến thuật toán mất hẳn khả năng nhảy đột biến để thoát cực tiểu cục bộ ở giai đoạn sau.
  - **Code lại để fit**: Cập nhật hàm `_dynamic_parameters` trong `gpu_engine.py` về đúng công thức tuyến tính kẹp ngưỡng $[0.5, 0.15]$ của baseline:
    ```python
    ratio = min(max(float(iter_idx) / float(max_iter), 0.0), 1.0)
    a = 2.0 - 2.0 * ratio
    p_hybrid = max(0.15, 0.5 * (1.0 - ratio))
    ```

---

### 2. Chuẩn hóa Sai số So sánh `PRECISION` (0.001 vs 0.0001)
- **Hiện trạng trong Baseline**:
  - `config.py`: `PRECISION = 0.001`.
  - Trong so sánh cải thiện chi phí: `if delta < -0.001:` hoặc `if delta <= PRECISION:`.
- **Hiện trạng trong GPU** (`gpu_kernels.py`):
  - Rất nhiều toán tử láng giềng đang hardcode `1e-4` ($0.0001$):
    - `if new_d < old_d - 1e-4:`
    - `if delta_d < -1e-4:`
    - `if (d1 + d2) < (old_d1 + old_d2) - 1e-4:`
    - `if cost_rem + cost_ins < -1e-4:`
  - Trong khi điều kiện Metropolis lại dùng: `if delta <= 1e-3:`.
- **Đánh giá & Giải pháp**:
  - Việc dùng $1e-4$ khắt khe hơn 10 lần so với baseline ($1e-3$). Trên số thực dấu phẩy động 32-bit/64-bit, sai lệch $1e-4$ có thể khiến một số cải tiến hợp lệ biên (borderline improvements) bị từ chối bỏ qua.
  - **Code lại để fit**: Thay toàn bộ các ngưỡng `-1e-4` trong `gpu_kernels.py` thành `-1e-3` (`-0.001`) để hoàn toàn đồng bộ với `PRECISION = 0.001` của baseline.

---

### 3. Tham số cấu hình SA Warmup
- **Hiện trạng trong Baseline** (`src_python_baseline/data.py`):
  - `sa_t0 = 100.0`
  - `sa_alpha = 0.95`
  - `sa_tmin = 0.1`
  - `sa_itermax = 100`
- **Hiện trạng trong GPU** (`src_python_gpu_SA_RCRS_GRASP/gpu_kernels.py:448`):
  - `temp = 100.0`
  - `cooling = 0.85` (thay vì 0.95)
  - `temp > 0.5` (thay vì 0.1)
  - `sa_iters = 25` (thay vì 100)
- **Đánh giá & Giải pháp**:
  - Trong baseline, SA chạy trên 1 luồng CPU cho số ít cá thể nên có thể chạy hàng trăm bước lặp. Trên GPU, bước khởi tạo thực hiện đồng thời cho tất cả các luồng; nếu để `itermax=100` và `cooling=0.95`, số bước lặp sẽ tăng gấp $\approx 15$ lần, khiến pha `CPU_PREP` khởi tạo ban đầu bị kéo dài thêm 15-20 giây.
  - **Đề xuất**: Đây là sự điều chỉnh có chủ đích hợp lý để đảm bảo tốc độ GPU. Tuy nhiên, nếu bạn muốn khởi tạo kỹ hơn tiệm cận baseline, có thể cân nhắc tăng nhẹ `sa_iters = 50` và `cooling = 0.90`.

---

### 4. Chiều dài chuỗi Or-Opt và 2-Exchange
- **Hiện trạng trong Baseline**:
  - `DEFAULT_OR_OPT_LEN = 3`
  - `DEFAUTL_EX_LEN = 2`
- **Hiện trạng trong GPU**:
  - Khi kích hoạt `paper_flags` trong `data.py`: `or_opt_len` bị gán về `2`.
- **Đánh giá & Giải pháp**:
  - Chuỗi Or-opt độ dài 3 cho phép di chuyển các đoạn gồm 3 khách hàng liên tiếp, có khả năng tái cấu trúc tuyến mạnh hơn chuỗi độ dài 2.
  - **Code lại để fit**: Trả lại `or_opt_len = 3` trong `data.py` để khớp với `DEFAULT_OR_OPT_LEN = 3`.

---

## TỔNG KẾT TRẠNG THÁI TRIỂN KHAI

1. **Các cải tiến cốt lõi (Khuyên giữ nguyên)**:
   - Kiến trúc 100% Full-GPU Resident, Zero Host Sync.
   - Lexicographic Acceptance (NV ưu tiên tuyệt đối, chia nhiệt độ theo TD).
   - Island Model & Ring Migration.
   - Greedy Best-Insertion (không mở tuyến ngẫu nhiên).
   - Nới lỏng tạm thời `V_NUM_RELAX = 3` trong lúc khám phá và kiểm tra hợp lệ nghiêm ngặt khi xuất nghiệm.

2. **4 điểm đã được code lại để fit hoàn toàn với Baseline (ĐÃ HOÀN TẤT)**:
   - ✅ **Điểm 1 - $p_{hybrid}$ (Sàn 0.15)**: Đã cập nhật `_dynamic_parameters` trong `gpu_engine.py` về đúng công thức tuyến tính của baseline: `p_hybrid = max(0.15, 0.5 * (1.0 - ratio))` với sàn chặn cứng $0.15$.
   - ✅ **Điểm 2 - Chuẩn hóa `PRECISION = 0.001`**: Đã thay thế toàn bộ các ngưỡng so sánh `1e-4` thành `1e-3` (`0.001`) trong `gpu_kernels.py`, `operator.py`, `search_framework.py`, đồng bộ 100% với `config.PRECISION`.
   - ✅ **Điểm 3 - SA Warmup**: Đã cập nhật `temp = 100.0`, `cooling = 0.95`, `temp > 0.1`, `sa_iters = 100` (`DEFAULT_SA_ITERATIONS = 100`) khớp hoàn toàn với `sa_t0`, `sa_alpha`, `sa_tmin`, `sa_itermax` của baseline.
   - ✅ **Điểm 4 - Or-Opt Length**: Đã cập nhật `self.or_opt_len = 3` trong `data.py` khớp với `DEFAULT_OR_OPT_LEN = 3` của baseline.
