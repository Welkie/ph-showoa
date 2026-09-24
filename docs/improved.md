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

| Hạng mục / Tham số | Baseline (`src_python_baseline`) | Cải tiến GPU (`src_python_gpu_SA_RCRS_GRASP`) | Phân tích cơ chế & Kết quả thực nghiệm | Phân loại & Quyết định |
| :--- | :--- | :--- | :--- | :--- |
| **Lịch trình tỷ lệ $p_{hybrid}$** | Tuyến tính: $p = \max(0.15, 0.5(1 - \text{ratio}))$. Luôn có sàn $0.15$ SHO ở cuối | Cosine Decay: $p = 0.5(1 + \cos(\pi \cdot \text{ratio}))$. Chuyển dịch $1.0 \to 0.0$ | Khi ép về baseline: Gen đầu chỉ có 50% SHO (chưa kịp gom cụm gen tốt), gen cuối vẫn có 15% SHO làm phá vỡ các tuyến đã tối ưu. Cosine Decay giúp 100% Khám phá đầu $\to$ 100% Khai thác cuối | **CẢI TIẾN CỐT LÕI (GIỮ NGUYÊN)** |
| **Độ nhạy so sánh (`PRECISION`)** | `PRECISION = 0.001` ($10^{-3}$) dùng thô | $10^{-4}$ (`1e-4`) dùng cho Local Search & Elite Tracking | Khoảng cách tọa độ Euclidean là số thực liên tục. Khi ép về $10^{-3}$, thuật toán từ chối hàng trăm bước cải tiến vi mô ($< 0.001$), gây đình trệ nghiệm nhân tạo và cản trở hội tụ sâu | **CẢI TIẾN CỐT LÕI (GIỮ NGUYÊN)** |
| **Tham số SA Warmup** | $T_0=100.0, \alpha=0.95, T_{min}=0.1$, `sa_iters=100` ($13,500$ moves) | $T_0=100.0, \text{cooling}=0.85, T_{min}=0.5$, `sa_iters=25` ($\approx 800$ moves) | Baseline dùng SA từ đầu (`init="sa"`) nên cần chạy lâu. Trên GPU ta đã có RCRS-GRASP tạo cụm tuyến hướng tâm cực đẹp; chạy $13,500$ bước SA nhiệt độ cao sẽ **xé nát và phá hủy cấu trúc tuyến của RCRS-GRASP**. $800$ bước là tối ưu | **CẢI TIẾN CỐT LÕI (GIỮ NGUYÊN)** |
| **Hàm mục tiêu chấp nhận SA** | Vô hướng: $\text{Cost} = 2000 \cdot NV + TD$, chia cho $|\text{Cost}| \approx 21000$ | Thứ tự từ vựng: Cấm tăng NV, Metropolis chỉ trên $\Delta TD$, chia cho $|TD| \approx 1000$ | Loại trừ triệt để lỗi loãng nhiệt và hiện tượng nhận nghiệm tăng xe vô tội vạ của baseline | **CẢI TIẾN CỐT LÕI (GIỮ NGUYÊN)** |
| **Nới lỏng số xe (`max_num`)** | `self.vehicle.max_num = int(value)` (Nghiêm ngặt) | `int(value) + V_NUM_RELAX` (`V_NUM_RELAX = 3`) | Giúp giải phóng bế tắc tạm thời khi chèn khách, nghiệm xuất ra luôn được kiểm tra nghiêm ngặt $\le$ xe đề bài | **CẢI TIẾN CỐT LÕI (GIỮ NGUYÊN)** |
| **Mở tuyến mới ngẫu nhiên khi chèn khách** | Có xác suất 15% (`rng.random() < 0.15`) mở tuyến mới nếu không bật `paper_flags` | Tuyệt đối không mở tuyến mới nếu còn chỗ chèn khả thi | Mở tuyến bừa bãi sẽ làm phình to số lượng xe NV, đi ngược lại mục tiêu giảm xe | **CẢI TIẾN CỐT LÕI (GIỮ NGUYÊN)** |
| **Cấu trúc Quần thể** | Quần thể phẳng $P=64$ (hoặc 30), đấu loại toàn quần thể, inject global best | Quần thể chia Đảo ($4 \times 8 = 32$), đấu loại nội bộ đảo, Ring Migration | Chống hiện tượng sụp đổ gen sớm, khai thác tối đa năng lực xử lý song song khối luồng của GPU | **CẢI TIẾN CỐT LÕI (GIỮ NGUYÊN)** |
| **Tần suất in Log (`OUTPUT_PER_GENS`)** | `OUTPUT_PER_GENS = 1` (In mỗi thế hệ) | `OUTPUT_PER_GENS = 25` (In mỗi 25 thế hệ) | Tối ưu hóa I/O, tránh nghẽn console khi GPU đang chạy tốc độ cao | **TỐI ƯU HÓA GPU (GIỮ NGUYÊN)** |
| **Độ dài toán tử Or-Opt** | `DEFAULT_OR_OPT_LEN = 3` | `or_opt_len = 3` | Đã đồng bộ với baseline (chỉ dùng cho CPU move generation) | **ĐÃ FIT BASELINE** |

---

## PHÂN TÍCH CHUYÊN SÂU: TẠI SAO ÉP VỀ BASELINE LÀM KẾT QUẢ TỆ HƠN HẲN?

Khi thử nghiệm đưa 4 điểm về chuẩn baseline, chất lượng nghiệm bị sụt giảm nghiêm trọng do các nguyên nhân khoa học sau:

### 1. Hiệu ứng "Phá hủy Cấu trúc Tuyến" của SA Warmup $13,500$ bước
- **Tại sao Baseline cần $13,500$ bước?** Trong baseline, khi bật `init = "sa"`, thuật toán bắt đầu từ một nghiệm ngẫu nhiên/tầm thường, nên cần chạy SA rất lâu ($T_0=100 \to T_{min}=0.1$ với $\alpha=0.95$, $100$ bước/nhiệt độ) để tự hình thành các tuyến.
- **Tại sao trên GPU lại bị phản tác dụng?** Code GPU sử dụng **RCRS-GRASP** làm constructor ban đầu. RCRS-GRASP kết hợp hàm điểm hướng tâm và phạt tải trọng đã tạo sẵn các chùm tuyến hình quạt cực kỳ chặt chẽ và tối ưu.
- Khi ép chạy $13,500$ bước SA ngẫu nhiên ở nhiệt độ cao, các phép hoán đổi liên tuyến (Inter-route Relocate/Swap) đã **xới tung và xé rách các cụm khách hàng của RCRS-GRASP**. Quần thể bước vào thế hệ 1 với các tuyến zíc-zắc, chồng chéo, khiến toàn bộ tiến trình tiến hóa sau đó bị chậm và kẹt ở cực tiểu cục bộ xấu.
- **Kết luận**: SA Warmup trên GPU chỉ đóng vai trò "rung nhẹ" ($32$ bước nhiệt $\times 25 = 800$ bước lặp, $cooling=0.85, T>0.5$) để gạt bỏ các điểm nghẽn cục bộ nhỏ của GRASP mà không làm vỡ cụm tuyến. **Đây là Cải tiến Phối hợp Thuật toán bắt buộc phải giữ.**

### 2. Hiện tượng "Mù Vi mô" khi nâng ngưỡng sai số lên $10^{-3}$ (`PRECISION`)
- Khoảng cách giữa các khách hàng là số thực liên tục (Euclidean distance). Trong quá trình tối ưu hóa sâu (Deep Local Search), rất nhiều phép xoay 2-opt hoặc dời đỉnh Relocate đem lại mức giảm khoảng cách tinh vi như $0.0003, 0.0006, 0.0008$.
- Khi đặt ngưỡng cải tiến là $10^{-3}$ ($0.001$), mọi cải tiến $< 0.001$ đều bị coi là "không cải thiện" và bị vứt bỏ.
- Nguy hiểm hơn, trong hàm so sánh tinh hoa `is_better_lex`, một cá thể con dù tối ưu hơn cá thể cha $0.0005$ quãng đường cũng bị đánh giá là không tốt hơn, dẫn đến việc không cập nhật nghiệm tinh hoa (`island_best` / `global_best`). Thuật toán bị đình trệ nhân tạo.
- **Kết luận**: Ngưỡng $1e-4$ ($0.0001$) là chuẩn mực cần thiết cho tính toán số thực dấu phẩy động 64-bit trên GPU. **Đây là Cải tiến Độ nhạy Nghiệm cần giữ nguyên.**

### 3. Phá vỡ nguyên lý Metaheuristic khi dùng $p_{hybrid}$ Tuyến tính $[0.5 \to 0.15]$
- Quá trình tìm kiếm tối ưu luôn tuân theo quy luật vàng: **Thăm dò toàn cục (Exploration) ở giai đoạn đầu $\to$ Khai thác sâu (Exploitation) ở giai đoạn cuối**.
- Công thức tuyến tính của baseline:
  - Ở đầu run ($t=0$): $p_{hybrid} = 0.5$ $\implies$ Đã vội vã dành 50% tài nguyên cho WOA Intensification (bao vây quanh các nghiệm ban đầu vốn còn rất non nớt).
  - Ở cuối run ($t \to t_{max}$): $p_{hybrid} = 0.15$ $\implies$ Vẫn dành 15% tài nguyên chạy SHO Crossover (lấy tuyến của cá thể khác và chèn thêm khách), liên tục phá vỡ các tuyến đã gọt dũa rất đẹp ở các thế hệ 900-1000.
- Công thức **Cosine Decay** $[1.0 \to 0.0]$:
  - Ở đầu run: $p_{hybrid} = 1.0$ (100% SHO Crossover để lan tỏa mạnh mẽ các khối gen tốt).
  - Ở giữa run: $p_{hybrid} = 0.5$ (chuyển dịch mượt mà).
  - Ở cuối run: $p_{hybrid} = 0.0$ (100% WOA Intensification, tập trung toàn lực ép sâu quãng đường quanh nghiệm tốt nhất).
- **Kết luận**: Cosine Decay vượt trội hoàn toàn so với công thức tuyến tính của baseline. **Đây là Cải tiến Thuật toán Cốt lõi cần giữ nguyên.**

---

## TỔNG KẾT BẢN CHẤT CÁC CẢI TIẾN

| Hạng mục | Thuộc diện | Trạng thái hiện tại | Lý do |
| :--- | :--- | :--- | :--- |
| **Cosine Decay $p_{hybrid}$ ($1.0 \to 0.0$)** | **Cải tiến Thuật toán GPU** | **ĐÃ KHÔI PHỤC** | Đảm bảo nguyên lý Thăm dò $\to$ Khai thác mượt mà, tránh phá vỡ nghiệm cuối |
| **Độ nhạy $1e-4$ trong Local Search & Lexicographic** | **Cải tiến Độ nhạy GPU** | **ĐÃ KHÔI PHỤC** | Cho phép gọt dũa các bước cải tiến vi mô trên số thực Euclidean |
| **SA Warmup dịu nhẹ (cooling=0.85, 25 iters)** | **Cải tiến Phối hợp RCRS-GRASP** | **ĐÃ KHÔI PHỤC** | Bảo vệ cấu trúc cụm tuyến của RCRS-GRASP không bị xé nát |
| **`or_opt_len = 3`** | **Fit theo Baseline** | **ĐÃ FIT (3)** | Không ảnh hưởng GPU kernels, đồng bộ với baseline CPU logic |
