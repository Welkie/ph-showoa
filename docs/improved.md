# Tổng hợp Các Cải tiến GPU trong `src_python_gpu_SA_RCRS_GRASP` so với Bài báo & Code Baseline (`PH-SHOWOA`)

Tài liệu này đối chiếu trực tiếp, khách quan và chính xác giữa phiên bản GPU (`src_python_gpu_SA_RCRS_GRASP`) với bài báo gốc **PH-SHOWOA** (*PLOS ONE 2026, doi: 10.1371/journal.pone.0343262*) và mã nguồn baseline (`src_python_baseline`).

Dưới đây là các cải tiến kỹ thuật cốt lõi hiện có của phiên bản GPU kèm phân tích nguyên lý giải thích **tại sao cải tiến GPU lại vượt trội hơn**:

---

### 1. Chuyển đổi Toàn diện sang Kiến trúc 100% Full-GPU Resident (Thay thế CPU Baseline)
* **So với Baseline (Paper PH-SHOWOA & Mã nguồn CPU Baseline)**:
  * *Baseline*: Thuật toán chạy 100% trên CPU (`--backends cpu`). Vòng lặp 1000 thế hệ được thực thi bằng đa tiến trình CPU (`ProcessPoolExecutor`) hoặc đơn tiến trình. Quá trình tính toán bị giới hạn nặng nề bởi năng lực tính toán tuần tự của CPU và chi phí IPC giữa các tiến trình, khiến thời gian chạy rất lâu (mất hàng giờ cho mỗi instance).
  * *Cải tiến GPU*: Chuyển đổi toàn bộ giải thuật thành **100% Full-GPU Resident** trên CUDA. Toàn bộ 1000 thế hệ cư trú và tính toán khép kín trên VRAM của GPU theo chu trình:
    `CPU_PREP` (nạp dữ liệu bài toán lên GPU 1 lần duy nhất) $\to$ `RUN_GPU_BEGIN` $\to$ **1000 thế hệ thuần GPU (chạy liên tục trên VRAM, không có bất kỳ lệnh đồng bộ `cuda.synchronize()` hay kéo dữ liệu về Host nào giữa các thế hệ)** $\to$ `RUN_GPU_END` $\to$ `CPU_DECODE` (chỉ copy nghiệm tốt nhất và log telemetry đúng 1 lần khi kết thúc run).
  * *Chi tiết hiện thực*:
    * *Bộ nhớ ping-pong*: cấp phát trước các mảng quần thể `current`, `next`, mảng ứng viên và mảng nháp một lần duy nhất, mỗi thế hệ đọc từ `current`, ghi vào `next` rồi hoán đổi con trỏ (zero-copy). Baseline thì liên tục cấp phát và hủy đối tượng `Solution`, `Route` trên heap Python, gây áp lực lên bộ gom rác (GC).
    * *RNG độc lập trên device*: mỗi cá thể có trạng thái Xorshift128 riêng trong VRAM, khởi tạo từ `seed + chỉ số cá thể` (mỗi run dùng một seed riêng), nên các luồng không tranh chấp và kết quả tái lập được khi cố định seed và backend.
* **Tại sao tốt hơn**:
  * Khai thác triệt để hàng nghìn nhân CUDA xử lý song song thay vì chỉ vài nhân CPU.
  * Tốc độ giải quyết 1000 thế hệ cho bài toán 100 khách hàng giảm từ hàng giờ trên CPU xuống chỉ còn khoảng ~120 giây trên card GPU phổ thông (Nvidia Tesla T4).
  * Vòng lặp thế hệ không cấp phát bộ nhớ mới (VRAM ổn định, không còn áp lực GC trong vòng lặp chính); chỉ bước giải mã nghiệm cuối mỗi run mới tạo đối tượng Python.

---

### 2. Thay thế Khởi tạo SA "Mù" bằng Quy trình SA_RCRS_GRASP kết hợp Ép Giảm Xe (Route Elimination)
* **So với Baseline (Paper PH-SHOWOA, Algorithm 2, Trang 11-12)**:
  * *Baseline*: Trong paper gốc, thuật toán khởi tạo là **Simulated Annealing thuần túy (Algorithm 2: The Initialization Procedure)**. Nó bắt đầu từ một nghiệm khởi tạo thô và chạy SA ngẫu nhiên rất lâu ($T_0=100.0, \alpha=0.95, T_{min}=0.1, itermax=100 \implies \approx 13.500$ phép thử ngẫu nhiên trên CPU). Quá trình này hoàn toàn là "thử sai ngẫu nhiên" (blind random perturbations), không có định hướng hình học không gian và hoàn toàn **không có cơ chế chủ động ép giảm số lượng xe**. Nghiệm ban đầu thường có số lượng xe lớn, làm lãng phí hàng trăm thế hệ tiến hóa sau đó chỉ để thu gọn xe.
  * *Cải tiến GPU*: Thay thế toàn bộ bằng quy trình khởi tạo 4 giai đoạn có định hướng chạy song song trên GPU (mỗi cá thể một thread):
    1. *RCRS-GRASP Construction*: Dựng các chùm tuyến hình quạt có định hướng rõ rệt thông qua danh sách ứng viên ngẫu nhiên RCL ($\alpha \in [\alpha_{lo}, \alpha_{hi}]$) kết hợp hàm phạt góc lệch hướng tâm ($rs\_pen$) và phạt đầy tải ($rc\_pen$).
    2. *SA Warmup Dịu nhẹ*: Sau khi đã có chùm tuyến đẹp từ GRASP, SA Warmup trên GPU chỉ đóng vai trò "rung lắc nhẹ" ($T_0=100.0, \text{cooling}=0.85, T_{min}=0.5, 25\text{ iters} \approx 800$ phép thử) để gọt dũa các nút giao cắt nhỏ mà **bảo toàn nguyên vẹn khung tuyến hình quạt** vừa dựng.
    3. *Route Elimination*: Thuật toán chuyên biệt quét tuần tự tìm và loại bỏ các tuyến ít khách, tái chèn khách sang các tuyến khác nhằm ép giảm số lượng xe ($NV$) ngay từ thế hệ 0 (5 lượt; Paper hoàn toàn không có bước này). Bước này còn được chạy lại trên toàn quần thể mỗi `local_search_interval` = 25 thế hệ, ngay trước Local Search định kỳ.
    4. *Deep Local Search*: Quét 2 lượt tối ưu hóa sâu để làm mịn quãng đường trước khi bước vào thế hệ 1.
* **Tại sao tốt hơn**:
  * Tạo ra các chùm tuyến hình quạt tối ưu ngay từ đầu thay vì trông cậy vào $13.500$ bước hoán đổi ngẫu nhiên kéo dài của baseline.
  * Ép số lượng xe $NV$ xuống mức tối thiểu ngay từ thế hệ 0 nhờ toán tử *Route Elimination*, giúp thuật toán tiết kiệm hàng trăm thế hệ tìm kiếm.

#### 2.1. Vì sao sa_rcrs_grasp tốt hơn SA thuần

Quần thể đầu vào của PH-SHOWOA cần đồng thời **chất lượng tốt** và **đa dạng cấu trúc**. SA thuần (Algorithm 2) khó đáp ứng cả hai, còn sa_rcrs_grasp thì có cơ chế riêng cho từng yêu cầu:

1. **Chất lượng và tính khả thi (RCRS)**
   * SA thuần chỉ rung quanh nghiệm thô bằng các phép dời ngẫu nhiên. Một phép dời hợp lệ về tải trọng vẫn có thể làm trễ dây chuyền các cửa sổ thời gian phía sau, nên nhiều phép thử bị loại và SA không có cơ chế chủ động giảm số xe.
   * RCRS xây tuyến bằng cách chèn từng khách, mỗi bước kiểm tra tải trọng và cửa sổ thời gian. Điểm chèn được chấm theo khoảng cách tăng thêm (TD), dung lượng dư (`rc_pen`) và độ lệch hướng tâm (`rs_pen`). Nhờ đó tuyến có dạng hình quạt, ít giao cắt, còn dư tải để nhận thêm khách, và số xe thấp ngay từ đầu.

2. **Đa dạng cấu trúc (GRASP)**
   * RCRS một mình là xác định: chạy 30 lần cho 30 nghiệm giống nhau. GRASP chọn ngẫu nhiên trong danh sách ứng viên RCL với α ∈ [0.10, 0.40] cho mỗi cá thể, nên 30 cá thể có cấu trúc tuyến khác nhau nhưng đều tốt.
   * SA thuần thì ngược lại: các phép dời nhỏ dễ đưa các cá thể về cùng một lưu vực, làm toán tử CrossoverGuided của SHO thiếu chất liệu khác biệt để khám phá.

3. **Các bước hoàn thiện (chỉ có trong sa_rcrs_grasp)**
   * *SA warmup dịu* chỉ gọt các giao cắt nhỏ mà giữ nguyên khung tuyến hình quạt.
   * *Route Elimination* chủ động gộp tuyến để ép giảm NV ngay từ thế hệ 0 (paper không có bước này).
   * *Deep LS* làm mịn quãng đường trước khi vào thế hệ 1.

**Kết quả mong đợi**: điểm xuất phát có NV thấp và TD tốt hơn, quần thể đa dạng hơn, nên vòng tiến hóa bớt phải dùng các thế hệ đầu để ép xe. Điều này được kiểm chứng trong mục 2.2.

#### 2.2. Ablation study (đo thực nghiệm)

**Mục tiêu**: chỉ so sánh **bước khởi tạo**, mọi thứ khác giữ nguyên.
* **(1) Full SA theo baseline**: mỗi khách một tuyến riêng (thứ tự xáo ngẫu nhiên), sau đó SA thuần theo paper ($T_0=100$, $\alpha=0.95$, $T_{min}=0.1$, itermax $=100$ trên cùng 4 toán tử lân cận của SA warmup: 2-opt trong tuyến, Relocate trong tuyến, Relocate giữa tuyến, Swap giữa tuyến). Không có GRASP, Route Elimination hay Deep LS lúc khởi tạo. Chạy trên bản sao tạm của kernel khởi tạo, không sửa code gốc.
* **(2) Code hiện tại** (`--init sa_rcrs_grasp`): RCRS-GRASP → SA warmup nhẹ → Route Elimination → Deep LS.
* **Giống nhau**: cùng engine (vòng SHO/WOA, island, migration, stagnation, Route Elimination + LS định kỳ mỗi 25 thế hệ), `--paper_flags`, `pop_size=30`, 6 đảo, `runs=30`, `max_iter` = 100 / 200 / 1000, cùng seed gốc, đo trên code hiện tại (hàm mục tiêu vô hướng, stagnation toàn cục), cùng `--compute_backend cpu` (Numba CPU, vì máy chạy không có GPU NVIDIA).

**Đường hội tụ (cost trung bình của cá thể tốt nhất, 30 runs, `max_iter=1000`; Δ = (2) − (1), $t$ thô)**:

| Instance | Thế hệ | (1) Full SA | (2) sa_rcrs_grasp | Δ | $t$ |
|---|---|---|---|---|---|
| cdp101 | 1 | 30258.5 (NV 14.00) | 25845.6 (NV 12.23) | −4412.9 | −18.4 |
| cdp101 | 25 | 26637.6 | 24673.4 | −1964.2 | −8.1 |
| cdp101 | 50 | 25201.6 | 23624.3 | −1577.3 | −9.5 |
| cdp101 | 100 | 23133.8 | 23028.6 | −105.2 | −1.6 |
| rcdp104 | 1 | 34356.9 (NV 15.93) | 26516.0 (NV 12.47) | −7840.9 | −22.4 |
| rcdp104 | 25 | 29319.1 | 26193.1 | −3126.0 | −12.5 |
| rcdp104 | 50 | 27627.2 | 25756.8 | −1870.4 | −9.0 |
| rcdp104 | 100 | 26366.5 | 25291.3 | −1075.2 | −5.1 |
| rcdp104 | 200 | 25211.1 | 24454.4 | −756.7 | −3.5 |
| rdp210 | 1 | 9448.2 (NV 3.97) | 7848.4 (NV 3.27) | −1599.8 | −9.3 |
| rdp210 | 25 | 7168.1 | 7122.9 | −45.2 | −3.5 |
| rdp210 | 50 | 7091.7 | 7064.1 | −27.6 | −3.3 |
| rdp210 | 100 | 7036.3 | 7028.7 | −7.5 | −1.3 |

**Kết quả cuối khi giới hạn số thế hệ (30 runs, cost TB ± độ lệch chuẩn)**:

| Instance | `max_iter` | (1) Full SA | (2) sa_rcrs_grasp | Δ | $t$ |
|---|---|---|---|---|---|
| cdp101 | 100 | 24282.7 ± 975.9 | 23177.7 ± 499.7 | −1105.0 | −5.5 |
| cdp101 | 200 | 23019.2 ± 29.7 | 23013.6 ± 21.7 | −5.6 | −0.8 |
| cdp101 | 1000 | 23001.2 ± 17.3 | 22995.1 ± 16.6 | −6.1 | −1.4 |
| rcdp104 | 100 | 26555.4 ± 1042.3 | 25192.5 ± 637.9 | −1362.9 | −6.1 |
| rcdp104 | 200 | 25039.3 ± 779.4 | 24301.6 ± 1034.6 | −737.7 | −3.1 |
| rcdp104 | 1000 | 23303.5 ± 369.5 | 23575.5 ± 776.4 | +272.0 | +1.7 |
| rdp210 | 100 | 7036.6 ± 23.7 | 7026.2 ± 19.5 | −10.4 | −1.9 |
| rdp210 | 200 | 7000.6 ± 17.8 | 7001.0 ± 20.4 | +0.4 | +0.1 |
| rdp210 | 1000 | 6985.1 ± 17.3 | 6987.3 ± 15.0 | +2.2 | +0.5 |

**Nhận xét**:
* **Điểm xuất phát**: sau khởi tạo (thế hệ 1), (2) có cost thấp hơn rất rõ ở cả 3 instance (thấp hơn khoảng 1600–7800, $t$ từ −9 đến −22) và ít xe hơn 0,7–3,5 xe (cdp101: 12.23 so với 14.00; rcdp104: 12.47 so với 15.93; rdp210: 3.27 so với 3.97).
* **Tốc độ hội tụ**: lợi thế kéo dài đáng kể trong 50–200 thế hệ đầu (cdp101 còn khác biệt có ý nghĩa đến thế hệ 50, rcdp104 đến thế hệ 200 với $t \approx -3{,}5$), rồi thu hẹp dần. Khi ngân sách thế hệ ngắn (`max_iter=100`), (2) tốt hơn rõ trên cdp101 và rcdp104 (cost thấp hơn khoảng 1100 và 1360, $t$ = −5,5 và −6,1); với `max_iter=200`, rcdp104 vẫn thấp hơn khoảng 740 ($t \approx -3{,}1$).
* **Ngân sách dài (`max_iter=1000`)**: hai cấu hình hội tụ về chất lượng gần nhau, không khác biệt có ý nghĩa (cdp101 $t = -1{,}4$; rdp210 $t = 0{,}5$). Riêng rcdp104, SA thuần có cost trung bình thấp hơn khoảng 272 ($t \approx 1{,}7$, Mann–Whitney hai phía $p \approx 0{,}53$) vì 29/30 run đạt 11 xe so với 25/30 của (2); khác biệt chưa có ý nghĩa thống kê.
* **Ngoại lệ**: rdp210 (3 xe, dễ) hội tụ rất nhanh nên lợi thế chỉ rõ ở khoảng 50 thế hệ đầu.
* **Kết luận**: sa_rcrs_grasp cho điểm xuất phát tốt hơn và tiết kiệm số thế hệ cần để đạt cùng chất lượng, rõ nhất khi ngân sách thế hệ ngắn và instance khó (nhiều xe). Với `max_iter=1000` không thấy lợi thế về chất lượng cuối (rcdp104 thậm chí hơi nghiêng về SA thuần, chưa có ý nghĩa).

* Thời gian: khởi tạo chỉ chạy 1 lần mỗi run nên chiếm phần nhỏ so với vòng tiến hóa; hai cấu hình có tổng thời gian chạy xấp xỉ nhau trên CPU (không phản ánh tốc độ GPU).
* **Giới hạn của ablation**: 3 instance, 30 runs; $t$ là ước lượng thô giả định phân phối gần chuẩn (cost nhảy theo bước 2000 khi đổi số xe). Chạy trên CPU, chưa chạy GPU/Kaggle. Phép so sánh là cả quy trình khởi tạo 5 giai đoạn, chưa tách riêng từng giai đoạn.

---

### 3. Mô hình Quần thể Đa Đảo (Island Model) & Di cư Vòng (Ring Migration)
* **So với Baseline (Paper PH-SHOWOA, Algorithm 1, Trang 10)**:
  * *Baseline*: Quần thể phẳng (single flat population). Theo Algorithm 1 của paper, nghiệm tốt nhất toàn cục `best` được lưu riêng (không chèn vào quần thể), nhưng **mọi cá thể đều dùng chung một `best` này làm hướng dẫn** khi cập nhật: SHO lai ghép lấy tuyến từ `best`, WOA bao vây và xoắn ốc quanh `best`. Vì cả quần thể cùng bị kéo về một nghiệm, các cá thể dễ hội tụ về cùng một hố cực tiểu cục bộ (hội tụ sớm).
  * *Cải tiến GPU*: Quần thể được chia thành các Đảo độc lập (mặc định $P=30$, 6 đảo, mỗi đảo 5 cá thể; nếu $P$ không chia hết cho số đảo thì tự chọn ước của $P$ gần số đảo yêu cầu nhất, hoặc 1 đảo nếu không có ước phù hợp, kèm cảnh báo `[WARN]`). Chọn cha mẹ (tournament 3 cá thể), nghiệm hướng dẫn cho SHO/WOA (best của đảo thay cho global best) và cập nhật nghiệm tốt nhất đều nằm trong nội bộ đảo. Các đảo chỉ trao đổi gen mỗi `migration_interval` = 20 thế hệ thông qua **Ring Migration** (best của đảo $i$ thay thế cá thể tệ nhất của đảo $i+1$, chỉ khi best đó có `Cost` thấp hơn).
* **Tại sao tốt hơn**:
  * Duy trì độ đa dạng di truyền lâu dài, chống hiện tượng sụp đổ gen sớm; khai thác tối đa tính song song của kiến trúc Warp/Block trên GPU.
* **Kiểm chứng (ablation quần thể phẳng so với đa đảo)**:
  * *Thiết lập*: cùng cấu hình (`sa_rcrs_grasp`, 30 runs, 1000 thế hệ, cùng seed gốc, CPU), chỉ đổi `--num_islands` (1 = quần thể phẳng). Cost = $2000 \cdot NV + TD$. Effect size $d$ = (cost TB 1 đảo − cost TB nhiều đảo) / độ lệch chuẩn gộp, $d > 0$ nghĩa là đa đảo tốt hơn; $p$ là Mann–Whitney một phía.

  **(a) $P=30$: 1 đảo so với 6 đảo trên 12 instance 100 khách** (cdp201 hội tụ cùng một nghiệm ở mọi run nên không có biến thiên):

    | Instance | 1 đảo: cost TB ± sd | 6 đảo: cost TB ± sd | Δ cost | $d$ | $p$ |
    |---|---|---|---|---|---|
    | cdp101 | 22990.6 ± 10.5 | 22995.1 ± 16.6 | +4.6 | −0.33 | 0.80 |
    | cdp103 | 21060.7 ± 62.5 | 21014.1 ± 59.8 | −46.6 | +0.76 | 0.003 |
    | rcdp101 | 33103.6 ± 936.8 | 33244.9 ± 854.3 | +141.2 | −0.16 | 0.79 |
    | rcdp104 | 23727.2 ± 871.0 | 23575.5 ± 776.4 | −151.7 | +0.18 | 0.030 |
    | rcdp202 | 9228.8 ± 46.0 | 9227.7 ± 26.0 | −1.1 | +0.03 | 0.57 |
    | rcdp205 | 9378.5 ± 36.1 | 9357.8 ± 28.6 | −20.8 | +0.64 | 0.011 |
    | rcdp207 | 8049.2 ± 977.3 | 7781.5 ± 915.8 | −267.6 | +0.28 | 0.058 |
    | rdp103 | 29198.7 ± 352.9 | 29252.0 ± 16.6 | +53.3 | −0.21 | 0.026 |
    | rdp201 | 9289.0 ± 20.4 | 9280.9 ± 15.3 | −8.0 | +0.44 | 0.077 |
    | rdp203 | 6989.8 ± 23.2 | 6975.0 ± 15.1 | −14.7 | +0.75 | 0.009 |
    | rdp210 | 6992.9 ± 19.9 | 6987.3 ± 15.0 | −5.6 | +0.32 | 0.12 |

    Ghi chú rdp103: cost trung bình của 1 đảo thấp hơn do 1 run duy nhất tìm được 13 xe; nhìn chung 6 đảo ổn định hơn nhiều (sd 16.6 so với 352.9).

  * 6 đảo có cost trung bình thấp hơn ở 8/12 instance, effect size trung bình $d \approx +0{,}25$ (nhỏ), giảm cost trung bình khoảng 0,4% (cao nhất 3,3% ở rcdp207). Kiểm định $t$ một mẫu trên các $d$ của 11 instance có biến thiên cho $p \approx 0{,}03$ (Wilcoxon $p \approx 0{,}04$).

  **(b) Số đảo và kích thước quần thể (5 instance: cdp103, rcdp205, rdp203, rcdp104, rcdp207; $d$ TB so với 1 đảo cùng $P$)**:

    | Cấu hình | $d$ trung bình | Số instance đa đảo tốt hơn | $p$ ($d>0$) |
    |---|---|---|---|
    | $P=30$, 3 đảo | +0.13 | 4/5 | 0.11 |
    | $P=30$, 6 đảo | +0.52 | 5/5 | 0.006 |
    | $P=30$, 10 đảo | +0.49 | 5/5 | 0.019 |
    | $P=60$, 12 đảo | +0.52 | 5/5 | 0.004 |

    Với $P=60$ và 12 đảo, đa đảo tốt hơn ở cả 5 instance (cdp103: −56.9, rcdp104: −157.9, rcdp207: −285.8, $p$ từ 0.002 đến 0.06). Với 3 đảo (đảo lớn, 10 cá thể) lợi thế gần như biến mất.
  * *Nhận xét*: đa đảo (từ 6 đảo trở lên) cho lợi thế nhỏ đến vừa nhưng nhất quán so với quần thể phẳng; lợi thế rõ hơn ở instance khó hoặc có nhiều nghiệm cạnh tranh (cdp103, rcdp205, rdp203, rcdp207) và khi quần thể lớn hơn; không có lợi thế về thời gian chạy (CPU: 357–484 s cho 30 runs ở cả hai cấu hình, vì số cá thể xử lý mỗi thế hệ không đổi). Với số đảo quá ít (3), mô hình không khác quần thể phẳng.
  * *Giới hạn*: 5 instance trong (b) được chọn sau khi thấy tín hiệu ở (a) nên có thiên lệch chọn; bằng chứng không thiên lệch là (a) trên toàn bộ 12 instance ($p \approx 0{,}03$, $d \approx 0{,}25$), mức nhỏ. Các $p$ đơn lẻ chưa hiệu chỉnh cho so sánh nhiều lần. Chạy trên CPU, chưa chạy GPU/Kaggle; hiệu quả riêng của migration (`migration_interval`) chưa được tách ra kiểm tra.

---|---|---|---|---|---|
    | cdp101 | 1 | 11.00 | 30/30 | 22990.6 ± 10.5 | 22979.1 |
    | cdp101 | 6 | 11.00 | 30/30 | 22995.1 ± 16.6 | 22977.6 |
    | rcdp104 | 1 | 11.23 | 23/30 | 23727.2 ± 871.0 | 23183.0 |
    | rcdp104 | 6 | 11.17 | 25/30 | 23575.5 ± 776.4 | 23186.8 |
    | rdp210 | 1 | 3.00 | 30/30 | 6992.9 ± 19.9 | 6956.9 |
    | rdp210 | 6 | 3.00 | 30/30 | 6987.3 ± 15.0 | 6963.3 |

  * *Nhận xét*: 6 đảo và 1 đảo cho kết quả gần như ngang nhau. Trên 2 instance dễ (cdp101, rdp210) chênh lệch cost nhỏ so với độ lệch chuẩn (+4.6 và −5.6, $t \approx 1{,}3$ và $-1{,}2$). Trên rcdp104 (khó), 6 đảo có cost trung bình thấp hơn khoảng 152 và nhỉnh hơn về số run đạt 11 xe (25/30 so với 23/30), nhưng $t \approx -0{,}7$ (Mann–Whitney một phía $p \approx 0{,}03$), tức hướng có lợi nhưng bằng chứng yếu.
  * *Giới hạn*: chỉ 3 instance, 30 runs; $t$ là ước lượng thô giả định phân phối gần chuẩn (cost của rcdp104 nhảy theo bước 2000 mỗi khi đổi số xe); chưa chạy trên GPU/Kaggle. Đây chưa phải bằng chứng đủ mạnh cho lợi thế của mô hình đa đảo; hiệu quả của riêng migration (`migration_interval`) và kích thước đảo chưa được tách ra kiểm tra.

---|---|---|---|---|---|
    | cdp101 | 1 | 11.00 | 30/30 | 22988.6 ± 22.4 | 22923.8 |
    | cdp101 | 6 | 11.00 | 30/30 | 22978.4 ± 24.7 | 22909.8 |
    | rcdp104 | 1 | 11.40 | 18/30 | 24065.7 ± 1018.3 | 23208.8 |
    | rcdp104 | 6 | 11.10 | 27/30 | 23438.7 ± 621.7 | 23130.5 |
    | rdp210 | 1 | 3.00 | 30/30 | 6981.5 ± 30.3 | 6910.7 |
    | rdp210 | 6 | 3.00 | 30/30 | 6915.1 ± 63.4 | 6735.5 |

  * *Nhận xét*: 6 đảo tốt hơn hoặc ngang 1 đảo ở cả 3 instance, không instance nào 1 đảo thắng. cdp101 (dễ) gần như hòa (cost thấp hơn khoảng 10, $t \approx -1.7$). rcdp104 (khó nhất) cho thấy rõ nhất: 27/30 run đạt 11 xe so với 18/30, cost trung bình thấp hơn khoảng 627 ($t \approx -2.9$). rdp210 cost trung bình thấp hơn khoảng 66 ($t \approx -5.2$) và tìm được nghiệm tốt nhất 6735.5, nhưng độ lệch chuẩn của 6 đảo lớn gấp đôi (63 so với 30), tức kết quả dao động nhiều hơn.
  * *Giới hạn*: chỉ 3 instance (cdp101 vốn dễ nên ít phân biệt), giá trị $t$ là ước lượng thô giả định phân phối gần chuẩn (cost của rcdp104 nhảy theo bước 2000 mỗi khi đổi số xe), và chưa chạy trên GPU/Kaggle. Đây là bằng chứng ủng hộ mô hình đa đảo chứ chưa phải kết luận chắc chắn; nên chạy lại trên nhiều instance hơn trước khi đưa vào bài báo. Hiệu quả của riêng migration (`migration_interval`) và kích thước đảo chưa được tách ra kiểm tra.

---

### 4. Thực thi Song song Toàn Quần thể cho Bộ Toán tử Láng giềng trên GPU
* **So với Baseline (Paper PH-SHOWOA, Algorithm 8, Trang 18 & Trang 21)**:
  * *Baseline*: Combined Local Search (2-opt, Relocate, Swap) chỉ áp dụng định kỳ cho **1 cá thể tốt nhất toàn cục** (*"restricts combined local search to periodic refinement of the global best solution"*). Mỗi move thử bằng `s.clone()` rất tốn kém.
  * *Cải tiến GPU*: Deep Local Search có 4 toán tử: 3 toán tử của baseline là 2-opt (trong tuyến), Relocate (trong tuyến và giữa tuyến, cài thành 2 vòng lặp riêng) và Swap (giữa hai tuyến), cộng thêm 1 toán tử mới **Inter 2-opt\*** (đổi đuôi hai tuyến). Chạy song song cho **toàn bộ $P$ cá thể** ở khởi tạo và định kỳ mỗi 25 thế hệ (cùng Route Elimination), thao tác trực tiếp trên mảng thay vì clone đối tượng. Với 3 toán tử giữa tuyến (Inter 2-opt\*, Relocate giữa tuyến, Swap), độ chênh lệch quãng đường $\Delta d$ được tính $O(1)$ bằng ma trận khoảng cách để lọc trước khi kiểm tra khả thi; 2 toán tử trong tuyến (2-opt, Relocate trong tuyến) đánh giá lại cả tuyến ($O(L)$) cho mỗi phép thử.
* **Tại sao tốt hơn**:
  * *Nâng cao chất lượng toàn quần thể*: Thay vì chỉ chăm chút 1 cá thể tinh hoa duy nhất, việc toàn bộ $P$ cá thể ở tất cả các đảo đều được tối ưu liên tục giúp kéo mặt bằng chất lượng nghiệm lên rất cao, cung cấp nguồn gen cha mẹ phong phú và ưu tú cho các phép lai ghép tiếp theo.
  * *Tốc độ vượt trội (Zero Object Allocation)*: Mọi phép thử đảo đoạn, hoán đổi khách hay dời tuyến đều thao tác trực tiếp trên mảng VRAM 3D `nodes[P, R, L]` (dùng mảng nháp), không cần `clone()` đối tượng như baseline; với các toán tử giữa tuyến, $\Delta d$ được lọc trước trong $O(1)$ qua ma trận `dist_matrix`. Triệt tiêu chi phí `clone()` đối tượng và bộ gom rác Garbage Collector của Python CPU.
* **Kiểm chứng (ablation)**:
  * *Thiết lập*: cùng cấu hình ($P=30$, 6 đảo, `sa_rcrs_grasp`, 30 runs, 1000 thế hệ, cùng seed gốc, CPU), 12 instance 100 khách. Cost = $2000 \cdot NV + TD$. So sánh code hiện tại (B: LS định kỳ cho toàn bộ $P$ cá thể, 5 toán tử) với hai biến thể, mỗi biến thể chỉ đổi đúng một yếu tố:
    * **(A1) LS chỉ cho cá thể tốt nhất** (như baseline): mỗi 25 thế hệ, Deep LS chỉ chạy cho 1 cá thể tốt nhất quần thể; Route Elimination định kỳ và LS lúc khởi tạo giữ nguyên.
    * **(A2) LS toàn quần thể nhưng bỏ Inter 2-opt\***: còn đúng 3 toán tử của baseline (2-opt, Relocate, Swap); trong code Relocate được cài thành 2 vòng riêng (trong tuyến và giữa tuyến) nên có 4 vòng lặp, nhưng cùng một tập lân cận với baseline.
  * Effect size $d$ = (cost TB biến thể − cost TB hiện tại) / độ lệch chuẩn gộp, $d > 0$ nghĩa là code hiện tại tốt hơn; $p$ là Mann–Whitney một phía (hiện tại < biến thể). cdp201 hội tụ cùng một nghiệm ở mọi cấu hình nên không có biến thiên.

    | Instance | (B) Hiện tại | (A1) LS chỉ cho best | $d$ | $p$ | (A2) Bỏ 2-opt\* | $d$ | $p$ |
    |---|---|---|---|---|---|---|---|
    | cdp101 | 22995.1 ± 16.6 | 23019.6 ± 34.4 | +0.91 | 0.001 | 23006.7 ± 19.5 | +0.64 | 0.005 |
    | cdp103 | 21014.1 ± 59.8 | 21144.2 ± 75.9 | +1.91 | <0.001 | 21174.4 ± 97.6 | +1.98 | <0.001 |
    | rcdp101 | 33244.9 ± 854.3 | 32985.5 ± 970.3 | −0.28 | 0.75 | 33392.3 ± 760.6 | +0.18 | 0.041 |
    | rcdp104 | 23575.5 ± 776.4 | 23662.5 ± 826.6 | +0.11 | 0.051 | 24016.7 ± 988.9 | +0.50 | <0.001 |
    | rcdp202 | 9227.7 ± 26.0 | 9283.8 ± 49.6 | +1.42 | <0.001 | 9282.2 ± 50.8 | +1.35 | <0.001 |
    | rcdp205 | 9357.8 ± 28.6 | 9432.8 ± 42.1 | +2.09 | <0.001 | 9429.8 ± 48.6 | +1.81 | <0.001 |
    | rcdp207 | 7781.5 ± 915.8 | 8333.2 ± 951.8 | +0.59 | <0.001 | 8061.0 ± 968.8 | +0.30 | 0.029 |
    | rdp103 | 29252.0 ± 16.6 | 29204.0 ± 351.2 | −0.19 | 0.005 | 29312.2 ± 40.3 | +1.96 | <0.001 |
    | rdp201 | 9280.9 ± 15.3 | 9309.1 ± 22.3 | +1.47 | <0.001 | 9308.4 ± 24.5 | +1.34 | <0.001 |
    | rdp203 | 6975.0 ± 15.1 | 7020.0 ± 36.9 | +1.59 | <0.001 | 6998.0 ± 20.6 | +1.27 | <0.001 |
    | rdp210 | 6987.3 ± 15.0 | 7035.9 ± 31.6 | +1.96 | <0.001 | 7005.4 ± 26.8 | +0.83 | 0.002 |

  * *Tổng hợp trên 11 instance có biến thiên*:

    | So sánh | $d$ trung bình | Hiện tại tốt hơn | $p$ ($t$ một mẫu) | $p$ (Wilcoxon) |
    |---|---|---|---|---|
    | LS toàn quần thể so với chỉ best (A1) | +1.05 (lớn) | 9/11 | 0.001 | 0.005 |
    | Thêm Inter 2-opt\* (so với A2) | +1.11 (lớn) | 11/11 | <0.001 | <0.001 |

  * *Nhận xét*:
    * Cả hai yếu tố đều cho lợi thế **lớn và nhất quán**, mạnh nhất trong các cải tiến đã kiểm chứng. Lợi thế có ngay từ thế hệ 100 (LS toàn quần thể: $d \approx 0{,}66$, 11/11 instance; Inter 2-opt\*: $d \approx 0{,}98$, 11/11) và tăng dần đến thế hệ 1000.
    * LS chỉ cho best làm tăng mạnh độ dao động giữa các run (độ lệch chuẩn thường gấp 1,5–2 lần), tức chất lượng toàn quần thể kém hơn.
    * Ngoại lệ của A1: rcdp101 (best-only nhỉnh hơn, không có ý nghĩa, $p = 0{,}75$) và rdp103 (cost TB của best-only thấp hơn do 1 run tìm được 13 xe, nhưng xét phân phối thì code hiện tại tốt hơn, $p = 0{,}005$).
    * Thời gian: đo tuần tự trên CPU (không chạy song song, đã trừ thời gian biên dịch JIT) không thấy khác biệt nhất quán (rdp210: 10,8 s/run so với 8,6 s/run cho best-only; rcdp104: 12,6 so với 15,5 s/run). Tuyên bố "tốc độ vượt trội nhờ không clone đối tượng" so với baseline CPU Python thuộc kiến trúc ở mục 1, không được đo trong ablation này.
  * *Giới hạn*: chạy trên CPU, chưa chạy GPU/Kaggle; $p$ đơn lẻ chưa hiệu chỉnh nhiều so sánh (nhưng phần lớn $p < 0{,}001$ nên vẫn có ý nghĩa sau hiệu chỉnh Bonferroni).
