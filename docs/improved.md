# Các cải tiến GPU trong `src_python_gpu_SA_RCRS_GRASP` so với bài báo và `src_python_baseline`

Tài liệu đối chiếu phiên bản GPU (`src_python_gpu_SA_RCRS_GRASP`) với bài báo **PH-SHOWOA** (*PLOS ONE 2026, doi: 10.1371/journal.pone.0343262*) và mã nguồn baseline (`src_python_baseline`, bản cài đặt sát paper trên CPU).

## 0. Phần nền chung với baseline

Bản GPU có cùng cấu trúc file với baseline (`algorithm1_solver.py` … `algorithm10_repair.py`, `problem.py`, `solution.py`, `cli.py`) và giữ nguyên các phần sau:

* **Mô hình**: mục 3 của paper (Eq. 1–11), chi phí phân cấp $TC = 2000 \cdot NV + TD$. Tải đầu tuyến bằng tổng lượng giao $\le Q$, sau mỗi khách tải $= $ tải $- d_i + p_i \in [0, Q]$. Có kiểm tra giới hạn đội xe và giờ về kho. Nghiệm cuối của mỗi run được kiểm tra lại bằng `check_feasibility_canonical` của baseline.
* **Vòng lặp chính (Algorithm 1)**: $a = 2 - 2t/T_{max}$, $p_{Hybrid} = \max(0{,}15;\ 0{,}5(1 - t/T_{max}))$.
* **Chấp nhận nghiệm (hướng B, giống baseline hiện tại)**: Algorithm 4 (WOA) và Algorithm 5 (SHO) chỉ sinh ứng viên. Nếu ứng viên không khả thi thì được sửa (Algorithm 10); nếu vẫn không khả thi thì giữ nghiệm cũ. Ngược lại ứng viên được trả về cả khi xấu hơn. Algorithm 1 (dòng 17–25) mới quyết định: tốt hơn hoặc bằng thì nhận; xấu hơn thì nhận với xác suất $\exp\big(-(n_f - c_f)/(10^{-6} + T\,|c_f|)\big)$, $T = 1 - t/T_{max}$.
* **Toán tử**:
  * SHO: tournament $k = 3$ không lặp, CrossoverGuided, đột biến với $p_m = 0{,}35$, Feasible_Or_Repair.
  * WOA: bao vây / di chuyển ngẫu nhiên / xoắn ốc cho từng tuyến, rollback theo tuyến.
* **LS và trì trệ**:
  * LS định kỳ mỗi 25 thế hệ.
  * Trì trệ: 50 thế hệ với ngưỡng $10^{-4}$, rồi DIVERSIFY (5 lần đột biến cho 20% cá thể tệ nhất).
* **Tham số**: `pop_size = 30`, `max_iter = 1000`. Khi chạy `--init sa`, Algorithm 2 dùng $T_0 = 1$, $\alpha = 0{,}9$, $T_{min} = 0{,}01$, itermax $= 20$.

Khác biệt không tránh được là bộ sinh số ngẫu nhiên: Xorshift128 riêng cho từng cá thể trên GPU, so với `random.Random` của Python. Vì vậy cùng seed không cho cùng nghiệm như baseline, chỉ so sánh được theo thống kê.

Bốn cải tiến dưới đây là những chỗ khác có chủ đích, đều bật mặc định. Phần ablation ở cuối dùng cờ dòng lệnh để tắt từng cải tiến.

---

### 1. Chạy toàn bộ trên GPU (full GPU-resident)

* **Baseline**:
  * Python thuần trên CPU; mỗi nghiệm là một đối tượng `Solution`, và mỗi phép thử phải `copy()` cả nghiệm.
  * Kết quả hiện có (`docs/ph_showoa_canonical_results.csv`): 1950 run, tổng khoảng 105 giờ CPU. Mỗi run 1000 thế hệ trên instance 100 khách mất 50–520 s.
* **Bản GPU**:
  * Cả 1000 thế hệ chạy trên VRAM; dữ liệu bài toán chỉ nạp lên GPU một lần.
  * Mỗi bước của Algorithm 1 (cập nhật SHO/WOA, LS, di cư, cập nhật best, xử lý trì trệ) là một lần gọi kernel.
  * Giữa các thế hệ không đồng bộ và không chép dữ liệu về host. Cuối run chỉ chép về nghiệm tốt nhất và log hội tụ.
* **Tại sao bản GPU tốt hơn**:
  * Cả quần thể được xử lý song song trên GPU thay vì lần lượt trên CPU.
  * Vòng lặp không tốn thời gian chuyển dữ liệu qua lại giữa CPU và GPU.
  * Không tạo và hủy đối tượng Python, nên không bị bộ gom rác làm chậm.
* **Chi tiết cài đặt**:
  * *Bộ nhớ ping-pong*: mảng quần thể `pop`/`next`, mảng ứng viên và mảng nháp được cấp phát một lần. Mỗi thế hệ đọc từ `pop`, ghi vào `next`, rồi đổi vai hai mảng.
  * *RNG trên device*: mỗi cá thể có trạng thái Xorshift128 riêng, khởi tạo từ seed của run và chỉ số cá thể. Kết quả tái lập được khi cố định seed và backend.
  * *Một mã nguồn cho hai backend*: cùng mã Numba biên dịch thành kernel CUDA (`--compute_backend cuda`) hoặc thành hàm CPU (`--compute_backend cpu`).
  * Thời gian chạy đầy đủ trên T4 sẽ đo cùng lúc với ablation.

---

### 2. Khởi tạo SA_RCRS_GRASP kết hợp Route Elimination

* **Baseline (Algorithm 2)**:
  * Paper nhận nghiệm ban đầu $S_0$ làm đầu vào nhưng không nói cách dựng.
  * Baseline tự chọn cách dựng: xếp khách theo thời điểm đóng cửa sổ (cộng nhiễu $U(-10, 10)$), rồi chèn từng khách vào vị trí rẻ nhất còn khả thi.
  * Tiếp theo chạy SA với $T_0 = 1$, $\alpha = 0{,}9$, $T_{min} = 0{,}01$, 20 vòng ở mỗi nhiệt độ, tổng khoảng 900 phép thử ngẫu nhiên.
  * Không có bước nào chủ động giảm số xe.
* **Bản GPU** (`--init sa_rcrs_grasp`, mặc định): mỗi cá thể chạy lần lượt 4 bước.
  1. *RCRS-GRASP*: dựng tuyến bằng cách chèn từng khách, có kiểm tra tải và cửa sổ thời gian. Mỗi lần chèn, cá thể chọn ngẫu nhiên trong danh sách ứng viên tốt nhất (RCL).
  2. *SA warm-up nhẹ*: chỉnh nhỏ các tuyến vừa dựng.
  3. *Route Elimination*: gỡ khách khỏi tuyến ít khách và chèn sang tuyến khác để giảm số xe. Paper không có bước này.
  4. *Local search của mục 4*.
* **Tại sao bản GPU tốt hơn**:
  * *Điểm xuất phát tốt hơn*: tuyến được dựng có định hướng (hình quạt, ít giao cắt, còn dư tải), không phải rung ngẫu nhiên quanh một nghiệm thô. Vì vậy NV và TD ban đầu đều thấp hơn.
  * *Giảm số xe ngay từ đầu*: Route Elimination đã ép NV xuống trước thế hệ 1, nên vòng tiến hóa không phải tốn các thế hệ đầu để giảm xe.
  * *Quần thể đa dạng*: mỗi cá thể dùng một giá trị $\alpha$ riêng của GRASP, nên 30 cá thể có cấu trúc tuyến khác nhau, cho SHO nhiều chất liệu để lai ghép.
* **Chi tiết cài đặt**:
  * Điểm chèn của RCRS được chấm theo $\Delta TD + 0{,}5\, rc\_pen + 0{,}3\, rs\_pen$. Trong đó $rc\_pen$ phạt phần tải vượt 70% sức chứa, $rs\_pen$ phạt độ lệch so với hướng tâm. Mỗi cá thể dùng $\alpha$ ngẫu nhiên trong $[0{,}10;\ 0{,}40]$.
  * SA warm-up: $T_0 = 100$, hệ số làm nguội $0{,}85$, $T_{min} = 0{,}5$, 25 phép thử ở mỗi nhiệt độ, tổng khoảng 825 phép thử.
  * Route Elimination chạy 5 lượt. Bước này cũng chạy lại cho cả quần thể mỗi 25 thế hệ, ngay trước LS định kỳ.
  * `--init sa` chạy đúng Algorithm 2 như baseline, dùng làm đối chứng trong ablation.

---

### 3. Mô hình đa đảo với di cư vòng

* **Baseline (Algorithm 1)**:
  * Chỉ có một quần thể phẳng.
  * Mọi cá thể dùng chung nghiệm tốt nhất toàn cục `best` làm hướng dẫn: SHO lấy tuyến từ `best` để lai ghép, WOA bao vây và xoắn ốc quanh `best`.
* **Bản GPU**:
  * Quần thể được chia thành các đảo; mặc định $P = 30$ chia thành 6 đảo, mỗi đảo 5 cá thể.
  * Tournament, nghiệm hướng dẫn của SHO/WOA (là best của đảo) và việc cập nhật best đều diễn ra trong từng đảo.
  * Cứ mỗi 20 thế hệ, các đảo trao đổi nghiệm theo vòng: best của đảo $i$ thay cá thể tệ nhất của đảo $i+1$ nếu tốt hơn.
* **Tại sao bản GPU tốt hơn**:
  * Ở baseline, cả quần thể bị kéo về cùng một nghiệm nên dễ hội tụ sớm vào một cực tiểu cục bộ.
  * Khi có nhiều đảo, mỗi đảo đi theo nghiệm dẫn đường riêng, nên quần thể giữ được nhiều hướng tìm kiếm lâu hơn.
  * Di cư vòng vẫn đưa dần nghiệm tốt sang các đảo khác, nên các đảo không bị tách rời hoàn toàn.
* **Chi tiết cài đặt**:
  * Nếu $P$ không chia hết cho số đảo yêu cầu, chương trình chọn số đảo là ước của $P$ gần nhất với số yêu cầu (ít nhất 2 đảo, mỗi đảo ít nhất 3 cá thể; nếu không có thì dùng 1 đảo) và in cảnh báo `[WARN]`.
  * Mỗi đảo là một đoạn liên tiếp trong mảng quần thể, nên không cần sắp xếp lại dữ liệu.
  * Nghiệm trả về là best toàn cục, lấy trên tất cả các đảo.
  * `--num_islands 1` cho quần thể phẳng như baseline.

---

### 4. Local search cho toàn quần thể + Inter 2-opt\*, chạy song song trong block trên GPU

* **Baseline (Algorithm 8)**:
  * Combined Local Search gồm 2-opt trong tuyến, Relocate và Swap.
  * Dùng first improvement: gặp bước cải thiện đầu tiên thì áp dụng ngay, rồi quét lại từ đầu, cho tới khi không còn bước cải thiện.
  * Chỉ chạy định kỳ cho **một** nghiệm, là nghiệm tốt nhất toàn cục.
  * Mỗi phép thử chép cả nghiệm (`copy()`).
* **Bản GPU**:
  * LS chạy cho **cả quần thể**: lúc khởi tạo và mỗi 25 thế hệ, cả $P$ cá thể đều được cải thiện.
  * Thêm toán tử **Inter 2-opt\***: hoán đổi phần đuôi của hai tuyến.
  * **Mỗi cá thể là một block 128 luồng**. Ở mỗi bước, các luồng cùng quét lân cận của nghiệm và block chọn ra bước tốt nhất (**best improvement**). Lặp lại cho tới khi không còn bước cải thiện.
* **Tại sao bản GPU tốt hơn**:
  * *Nâng chất lượng cả quần thể*: mọi cá thể đều được cải thiện định kỳ, nên SHO/WOA lai ghép và dẫn đường từ những nghiệm tốt hơn, chứ không chỉ dựa vào một cá thể tinh hoa.
  * *Lân cận rộng hơn*: Inter 2-opt\* nối lại các tuyến theo cách mà Relocate và Swap không làm được trong một bước.
  * *Dùng đúng sức song song của GPU*: hàng trăm phép thử của cùng một nghiệm được đánh giá cùng lúc, và 30 block chạy đồng thời trên các SM khác nhau. Đo trên Kaggle T4, chạy LS cho cả 30 cá thể mất 0,01–0,07 s. Cách mỗi cá thể một luồng mất 0,6–49 s; CPU chạy lần lượt với first improvement mất 0,03–1,6 s.
  * *Mỗi lần LS cho nghiệm tốt hơn*: trên cả 5 instance đã đo, chọn bước tốt nhất ở mỗi bước cho cost sau LS thấp hơn first improvement.
* **Chi tiết cài đặt**:
  * Lân cận có 4 loại bước: 2-opt trong tuyến, Relocate (trong tuyến và giữa tuyến), Swap giữa hai tuyến, Inter 2-opt\*. Mỗi phép thử được mã hóa thành một cặp vị trí (tuyến, vị trí) kèm loại bước; luồng $t$ xét các cặp $t, t + 128, \dots$
  * $\Delta TD$ của mọi bước tính trong $O(1)$ từ ma trận khoảng cách. Kiểm tra khả thi (tải, cửa sổ thời gian) chỉ chạy cho bước có cải thiện và tốt hơn bước tốt nhất hiện có của luồng đó. Khi kiểm tra, chương trình đi theo từng đoạn của tuyến cũ để duyệt tuyến mới, không chép tuyến.
  * Block chọn bước tốt nhất bằng phép rút gọn trong bộ nhớ chia sẻ:
    * ưu tiên bước làm rỗng một tuyến (giảm NV), sau đó đến bước có $\Delta TD$ nhỏ nhất;
    * nếu hai bước bằng nhau thì lấy bước có mã nhỏ hơn;
    * luồng 0 áp dụng bước được chọn.
  * Vì kết quả không phụ thuộc số luồng, backend CPU chạy cùng phép quét đó với 1 luồng và cho **cùng một nghiệm**. Đã kiểm tra trên CUDA simulator với một nghiệm cần 13 bước cải thiện: hai bản khớp từng phần tử.
  * Các điểm lệch so với paper:
    * LS chạy cho toàn quần thể thay vì chỉ cho best;
    * có thêm Inter 2-opt\*;
    * dùng best improvement thay vì first improvement.

    Điều kiện dừng giữ như Algorithm 8: dừng khi hết bước cải thiện.
  * Cách đo: 30 nghiệm thực tế, là quần thể ngay trước lần LS ở thế hệ 25 (seed 42). CPU là Xeon của phiên Kaggle. Không tính thời gian biên dịch và chép dữ liệu. Mã đo nằm trong `kaggle/ls_bench/`.

    | Instance | FI, CPU, 30 nghiệm lần lượt | FI, GPU, 1 luồng/cá thể | BI, CPU, 30 nghiệm lần lượt | **BI, GPU, 1 block/cá thể** | Cost TB sau LS: FI → BI |
    |---|---|---|---|---|---|
    | cdp101 | 0,32 s | 7,41 s | 0,16 s | **0,03 s** | 26036.0 → 25757.8 |
    | rcdp101 | 0,03 s | 0,62 s | 0,05 s | **0,01 s** | 36086.2 → 36084.7 |
    | rcdp104 | 0,59 s | 14,12 s | 0,43 s | **0,05 s** | 25625.0 → 25551.2 |
    | rcdp202 | 1,14 s | 33,03 s | 0,31 s | **0,07 s** | 10057.7 → 9983.6 |
    | rdp210 | 1,58 s | 48,65 s | 0,36 s | **0,07 s** | 9022.5 → 9014.4 |

    FI = first improvement (Algorithm 8), BI = best improvement (bản GPU).
  * Giới hạn: với **một** nghiệm đơn lẻ, BI trên GPU chưa nhanh hơn BI trên CPU. Ở rcdp104 hai bên ngang nhau; ở rcdp202 và rdp210, GPU chậm hơn khoảng 2,5 lần. Lợi thế của GPU là chạy LS cho cả quần thể trong khoảng thời gian của một nghiệm.

---

## Ablation (chưa chạy lại)

Các bảng ablation trước đây đo trên code cũ, trước khi đồng bộ với baseline: chấp nhận nghiệm kiểu khác, LS first improvement giới hạn 2 lượt, phá 40% quần thể khi trì trệ. Số liệu đó không còn áp dụng nên đã bỏ.

Lượt chạy ngày 24/10 trên code Direction B cùng LS first improvement cũng đã dừng giữa chừng, khi chuyển sang cách LS của mục 4. Code hiện tại sẽ được đo lại theo kế hoạch dưới đây.

**Thiết lập chung**:
* 30 run, seed 42–71, `pop_size = 30`, `max_iter = 1000`, 6 đảo, `--init sa_rcrs_grasp`, chạy trên Kaggle T4 (`--compute_backend cuda`).
* Mọi mục dùng chung 12 instance 100 khách, 2 instance cho mỗi nhóm của bộ Wang & Chen:

  | Nhóm | Đặc điểm | Instance |
  |---|---|---|
  | Cdp1 | khách theo cụm, cửa sổ thời gian hẹp | cdp101, cdp103 |
  | Cdp2 | khách theo cụm, cửa sổ rộng | cdp203, cdp207 |
  | Rdp1 | khách ngẫu nhiên, cửa sổ hẹp | rdp103, rdp107 |
  | Rdp2 | khách ngẫu nhiên, cửa sổ rộng | rdp201, rdp210 |
  | RCdp1 | trộn cụm và ngẫu nhiên, cửa sổ hẹp | rcdp101, rcdp104 |
  | RCdp2 | trộn cụm và ngẫu nhiên, cửa sổ rộng | rcdp202, rcdp207 |

  Không dùng instance nhỏ (5–50 khách) vì mọi biến thể đều dễ đạt cùng một nghiệm.
* Mỗi biến thể chỉ đổi đúng một yếu tố so với cấu hình đầy đủ (B).
* Thống kê:
  * cost TB ± độ lệch chuẩn và phân bố NV;
  * effect size $d$ = (cost TB biến thể − cost TB của B) / độ lệch chuẩn gộp, nên $d > 0$ nghĩa là B tốt hơn;
  * $p$ Mann–Whitney một phía cho từng instance;
  * tổng hợp bằng $t$ một mẫu và Wilcoxon trên các $d$;
  * thời gian mỗi run.

| Mục | Biến thể | Cờ |
|---|---|---|
| 2 | Khởi tạo Algorithm 2 so với SA_RCRS_GRASP, `max_iter` 100 / 200 / 1000, kèm đường hội tụ | `--init sa`, `--max_iter` |
| 3a | Quần thể phẳng | `--num_islands 1` |
| 3b | Số đảo và kích thước quần thể: $P=30$ với 3 / 10 đảo; $P=60$ với 12 đảo so với 1 đảo | `--num_islands`, `--pop_size 60` |
| 4 – A1 | LS định kỳ chỉ cho cá thể tốt nhất (như paper) | `--ls_scope best` |
| 4 – A2 | Bỏ Inter 2-opt\* | `--no_ls_2opt_star` |
| 4 – A3 | First improvement, mỗi cá thể một luồng (cách cũ) | `--ls_strategy first` |

Ý nghĩa các biến thể của mục 4 (mỗi biến thể tắt đúng một phần của cải tiến 4):

| Biến thể | LS cho ai | Inter 2-opt\* | Cách chọn bước | Trả lời câu hỏi |
|---|---|---|---|---|
| B (đầy đủ) | cả quần thể | có | best improvement, 1 block/cá thể | mốc so sánh |
| A1 | chỉ cá thể tốt nhất | có | best improvement | LS cho cả quần thể có tốt hơn chỉ cho best? |
| A2 | cả quần thể | không | best improvement | Inter 2-opt\* có giúp không? |
| A3 | cả quần thể | có | first improvement, 1 luồng/cá thể | Best improvement ảnh hưởng chất lượng và thời gian thế nào? |

Thứ tự chạy: mục 4 và 3a trước, rồi mục 2, cuối cùng 3b. Trước mỗi lần đẩy phiên, orchestrator kiểm tra quota GPU; nếu thiếu quota thì phần bị cắt là 3b.
