# Cải tiến GPU của PH-SHOWOA

So sánh bản GPU (`src_python_gpu_SA_RCRS_GRASP`) với bài báo PH-SHOWOA (*PLOS ONE 2026, doi: 10.1371/journal.pone.0343262*) và baseline canonical (`src_python_baseline`, cài đặt sát paper trên CPU). **fullcaitienGPU** là bản GPU bật cả 3 cải tiến (cấu hình mặc định).

## 0. Giữ nguyên từ paper

* Mô hình: mục 3 của paper (Eq. 1–11), cost phân cấp $TC = 2000 \cdot NV + TD$; nghiệm cuối mỗi run được kiểm tra lại bằng `check_feasibility_canonical` của baseline.
* Algorithm 1: $a = 2 - 2t/T_{max}$, $p_{Hybrid} = \max(0{,}15;\ 0{,}5(1 - t/T_{max}))$. WOA và SHO chỉ sinh ứng viên (sửa bằng Algorithm 10 nếu không khả thi); Algorithm 1 nhận ứng viên xấu hơn với xác suất $\exp\big(-(n_f - c_f)/(10^{-6} + T\,|c_f|)\big)$, $T = 1 - t/T_{max}$.
* SHO: tournament $k = 3$, CrossoverGuided, đột biến $p_m = 0{,}35$. WOA: bao vây / ngẫu nhiên / xoắn ốc theo từng tuyến.
* LS định kỳ mỗi 25 thế hệ; trì trệ 50 thế hệ (ngưỡng $10^{-4}$) thì DIVERSIFY (5 đột biến cho 20% cá thể tệ nhất).
* `pop_size = 30`, `max_iter = 1000`.
* Khác biệt không tránh được: RNG là Xorshift128 riêng cho mỗi cá thể, nên cùng seed không cho cùng nghiệm như baseline; chỉ so sánh theo thống kê.

---

### 1. Chạy toàn bộ trên GPU, mỗi cá thể một block

* **Paper / baseline**: Python một luồng trên CPU; mỗi phép thử chép cả nghiệm rồi đi lại cả tuyến. Một run trên instance 100 khách mất 47–501 s.
* **Cải tiến**:
  * Cả 1000 thế hệ chạy trên GPU; mỗi bước của Algorithm 1 là một kernel, không chép dữ liệu về host giữa các thế hệ.
  * Mỗi cá thể là một block 64 thread. Thread 0 làm phần ngẫu nhiên (WOA, SHO, chấp nhận nghiệm) theo đúng thứ tự rút số. Cả block làm các vòng quét (repair, chèn khách khi khởi tạo, route elimination): mỗi thread đánh giá một phần ứng viên, rồi rút gọn trong block lấy ứng viên tốt nhất.
  * Đánh giá ứng viên không ghi ra bộ nhớ: đi trên tuyến như thể đã chèn / đảo. Bảng tiền tố (thời gian, quãng đường, tải sau từng vị trí) trong shared memory cho phép chỉ đi phần tuyến sau vị trí thay đổi.
  * Tính đúng: phép tính giữ đúng thứ tự như khi chạy tuần tự và hòa thì chọn chỉ số nhỏ nhất, nên song song hóa không đổi nghiệm (đã kiểm tra trên CUDA simulator: tuyến, NV, TD, đường hội tụ trùng nhau).
* **Kết quả** (12 instance, 30 run, seed 42–71; GPU: Kaggle T4). Thời gian đo như baseline: mỗi run từ đầu `solve()` tới khi có nghiệm tốt nhất; cả hai bên không tính phần nạp instance (trước `solve()`) và bước `check_feasibility_canonical` cuối. Bản GPU còn tính thêm các chi phí chỉ GPU mới có: nạp dữ liệu lên GPU và cấp phát (mỗi run), biên dịch JIT 60–85 s (30 run trong một tiến trình nên JIT chỉ nằm trong run 1). Vì vậy tỉ lệ tăng tốc là cận dưới. Bảng ghi trung bình 30 run. "GPU, `--init sa`" là bản GPU khởi tạo bằng Algorithm 2 như paper nhưng vẫn bật mục 3 và route elimination định kỳ (mục 2), nên không phải đúng thuật toán của paper. Trong ngoặc là phân bố NV (số xe: số run) khi NV thay đổi giữa các run.

    | Instance | Baseline | GPU, `--init sa` | fullcaitienGPU | Baseline / fullcaitienGPU | Baseline: cost TB ± sd | fullcaitienGPU: cost TB ± sd |
    |---|---|---|---|---|---|---|
    | cdp101 | 47 s | 5,4 s | 13,1 s | 3,6× | 24121.5 ± 976.7 (11: 14, 12: 16) | **23001.1 ± 12.2** |
    | cdp103 | 50 s | 4,7 s | 12,0 s | 4,2× | 22561.3 ± 851.1 (10: 12, 11: 18) | **20965.6 ± 42.3** |
    | cdp203 | 83 s | 4,7 s | 73,3 s | 1,1× | 6668.5 ± 44.7 | **6593.6 ± 4.1** |
    | cdp207 | 123 s | 5,9 s | 59,7 s | 2,1× | 6735.7 ± 16.3 | **6588.3 ± 0.0** |
    | rdp103 | 155 s | 11,3 s | 21,9 s | 7,1× | 27727.0 ± 715.6 (13: 25, 14: 5) | 28728.3 ± 847.9 (13: 8, 14: 22) |
    | rdp107 | 136 s | 11,0 s | 22,9 s | 6,0× | 25240.5 ± 47.9 | **22978.9 ± 486.9** (10: 2, 11: 28) |
    | rdp201 | 450 s | 13,0 s | 73,4 s | 6,1× | 9406.8 ± 51.0 | **9260.8 ± 7.5** |
    | rdp210 | 455 s | 13,9 s | 129,4 s | 3,5× | 7130.7 ± 39.7 | **6967.6 ± 7.1** |
    | rcdp101 | 137 s | 10,1 s | 17,4 s | 7,8× | 36405.0 ± 1080.7 (16: 1, 17: 19, 18: 10) | **31481.3 ± 581.8** (14: 3, 15: 27) |
    | rcdp104 | 123 s | 9,7 s | 19,2 s | 6,4× | 26189.3 ± 1488.6 (11: 3, 12: 13, 13: 13, 14: 1) | **21237.0 ± 23.4** |
    | rcdp202 | 396 s | 13,0 s | 83,8 s | 4,7× | 9479.3 ± 65.6 | **7926.0 ± 699.6** (3: 23, 4: 7) |
    | rcdp207 | 501 s | 13,0 s | 111,4 s | 4,5× | 9178.5 ± 364.0 (3: 1, 4: 29) | **7134.6 ± 32.2** |
    | **Tổng** | **2656 s** | **116 s** | **638 s** | **4,2×** | | |

  * Khởi tạo như paper (`--init sa`), bản GPU nhanh hơn baseline 9–39 lần (tổng 23 lần), nhiều nhất ở nhóm loại 2 (tuyến dài). Con số này gồm cả mục 3 và route elimination định kỳ, nên không phải tăng tốc riêng của mục 1 (chưa chạy cấu hình GPU đúng thuật toán paper). fullcaitienGPU chậm hơn bản đó 5,5 lần vì Ejection Pool (mục 2) nhưng vẫn nhanh hơn baseline 4,2 lần.
  * fullcaitienGPU có cost thấp hơn baseline ở 11/12 instance (Mann–Whitney $p < 10^{-9}$) và ổn định hơn nhiều; rdp103 kém hơn nhưng không có ý nghĩa ($p = 0{,}13$, nguyên nhân ở mục 2).
  * Hai bên chạy trên phần cứng khác nhau (CPU một luồng so với T4), nên tỉ lệ thời gian mang tính tham khảo.

---

### 2. Khởi tạo RCRS-GRASP + Route Elimination + Ejection Pool

* **Paper / baseline (Algorithm 2)**: paper không nói cách dựng $S_0$; baseline chèn khách theo thời điểm đóng cửa sổ (cộng nhiễu) vào vị trí rẻ nhất, rồi chạy SA ($T_0 = 1$, $\alpha = 0{,}9$, $T_{min} = 0{,}01$, 20 vòng mỗi nhiệt độ). Không có bước nào chủ động giảm số xe.
* **Quan sát**: NV của cá thể tốt nhất ở thế hệ 1 gần như bằng NV cuối cùng; vòng tiến hóa hầu như không bớt được xe. Vì mỗi xe đáng 2000, lợi lớn nhất nằm ở việc giảm NV ngay khi khởi tạo.
* **Cải tiến** (mỗi cá thể, không dùng SA):
  1. *RCRS-GRASP*: chèn từng khách, chọn ngẫu nhiên trong danh sách ứng viên tốt nhất (RCL). Điểm chèn $\Delta TD + 0{,}5\, rc\_pen + 0{,}3\, rs\_pen$ ($rc\_pen$ phạt tải vượt 70% sức chứa, $rs\_pen$ phạt lệch hướng tâm); $\alpha \in [0{,}10;\ 0{,}40]$ ngẫu nhiên cho mỗi cá thể nên quần thể đa dạng.
  2. *Route Elimination*: chuyển khách của tuyến ít khách sang tuyến khác (5 lượt).
  3. *Ejection Pool* (Nagata & Bräysy 2009, rút gọn, không có squeeze và perturbation): xóa một tuyến ngẫu nhiên, đưa khách vào ngăn xếp rồi chèn lại từng khách vào vị trí khả thi có $\Delta TD$ nhỏ nhất. Không có vị trí khả thi thì chèn bằng cách đẩy ra 1 khách (2 nếu 1 không được) của tuyến nhận, chọn theo tổng bộ đếm phạt rồi $\Delta TD$; khách bị đẩy quay lại ngăn xếp. Nhờ đổi chỗ khách giữa các tuyến nên bớt được xe cả khi các tuyến đã gần kín.
  4. LS của mục 3.
  * Trong vòng lặp chính, Route Elimination (5 lượt) còn chạy cho mọi cá thể mỗi 25 thế hệ, ngay trước LS định kỳ. Bước này chạy với mọi kiểu khởi tạo, kể cả `--init sa`. Baseline không có bước này.
  * Tham số Ejection Pool: tổng 5000 lần lấy khách khỏi ngăn xếp; một lần xóa tuyến thất bại sau 300 lần lấy (khôi phục nghiệm cũ); dừng sau 10 lần thất bại liên tiếp. Các lựa chọn (tuyến, vị trí, khách bị đẩy) chia cho 64 thread của block.
* **Chọn tham số Ejection Pool** (6 instance, 30 run; so với `--init sa`; thời gian là tổng giây/run của 6 instance, `sa` mất 44 s):

    | Cấu hình | Budget | Lấy tối đa / tuyến | Thất bại liên tiếp | Đẩy tối đa | Cost so với `sa` | Tốt hơn / kém hơn | rcdp104 10 xe | Thời gian |
    |---|---|---|---|---|---|---|---|---|
    | grasp_ep | 3000 | 300 | 2 | 2 | −2,94% | 4 / 0 | 14/30 | 104 s |
    | nocap | 3000 | 3000 | 2 | 2 | −3,15% | 3 / 0 | 19/30 | 273 s |
    | k1 | 5000 | 1000 | 3 | 1 | −3,28% | 5 / 0 | 19/30 | 283 s |
    | a1000 | 5000 | 1000 | 3 | 2 | −3,86% | 5 / 0 | 28/30 | 329 s |
    | **c300f10 (chọn)** | 5000 | 300 | 10 | 2 | **−3,88%** | **5 / 0** | **30/30** | 300 s |

* **Kết quả** (cột −mục 2 của bảng ablation):
  * Bỏ mục 2 làm cost tăng 3,19%; fullcaitienGPU tốt hơn có ý nghĩa ở 4/12 instance (rcdp101, rcdp104, rcdp202, rcdp207), kém ở 1 (rdp103); Wilcoxon $p = 0{,}042$.
  * Lợi đến từ NV: rcdp101 bớt 1–2 xe, rcdp104 bớt 1 xe ở mọi run, rcdp202 đạt 3 xe ở 23/30 run (Algorithm 2: 4 xe ở mọi run), rcdp207 đạt 3 xe ở mọi run.
  * Ngoại lệ rdp103: Algorithm 2 đạt 13 xe ở 30/30 run, fullcaitienGPU chỉ 8/30.
  * Đổi lại thời gian tăng 5,5 lần, nhiều nhất ở instance không bớt được xe (Ejection Pool thử đủ 10 lần thất bại mới dừng); độ lệch chuẩn lớn hơn ở 3 instance vì NV thay đổi giữa các run.

---

### 3. LS cho cả quần thể + Inter 2-opt\*, best improvement song song trong block

* **Paper / baseline (Algorithm 8)**: 2-opt trong tuyến, Relocate, Swap; first improvement; chỉ chạy cho nghiệm tốt nhất.
* **Cải tiến**:
  * LS cho **cả quần thể** (lúc khởi tạo và mỗi 25 thế hệ), nên SHO/WOA lai ghép từ những nghiệm tốt hơn.
  * Thêm **Inter 2-opt\*** (đổi phần đuôi của hai tuyến).
  * **Best improvement**, mỗi cá thể một block 128 thread: các thread cùng quét lân cận ($\Delta TD$ tính $O(1)$, chỉ kiểm tra khả thi cho bước có cải thiện), block chọn bước tốt nhất (ưu tiên bước làm rỗng tuyến, rồi $\Delta TD$ nhỏ nhất, hòa thì mã nhỏ nhất). Kết quả không phụ thuộc số thread.
* **Tốc độ một lần LS cho cả 30 cá thể** (quần thể thật ở thế hệ 25, Kaggle T4; FI = first improvement, BI = best improvement):

    | Instance | FI, CPU | FI, GPU, 1 thread/cá thể | BI, CPU | **BI, GPU, 1 block/cá thể** |
    |---|---|---|---|---|
    | cdp101 | 0,32 s | 7,41 s | 0,16 s | **0,03 s** |
    | rcdp104 | 0,59 s | 14,12 s | 0,43 s | **0,05 s** |
    | rcdp202 | 1,14 s | 33,03 s | 0,31 s | **0,07 s** |
    | rdp210 | 1,58 s | 48,65 s | 0,36 s | **0,07 s** |

    Nhờ vậy LS cho cả quần thể không làm chậm run: fullcaitienGPU 53 s/run so với 55 s/run khi tắt mục 3.
* **Kết quả** (cột −mục 3 của bảng ablation):
  * *TD thấp hơn khi cùng số xe*: ở 8 instance có cùng phân bố NV, fullcaitienGPU có cost (tức TD) thấp hơn có ý nghĩa ở 7, instance còn lại cho cùng nghiệm. Tính chung, tốt hơn có ý nghĩa ở 8/12 instance, không kém ở instance nào.
  * *Ổn định hơn*: độ lệch chuẩn nhỏ hơn có ý nghĩa ở 5/12 instance (rdp201: 7,5 so với 39,4; rdp210: 7,1 so với 33,1), không lớn hơn ở instance nào.
  * *Hạn chế*: khi tắt mục 3, LS bớt xe thường xuyên hơn ở rdp103 (13 xe ở 24/30 run so với 8/30), rdp107 (10 xe ở 14/30 so với 2/30), rcdp202 (3 xe ở 26/30 so với 23/30); không có ý nghĩa ở từng instance, nhưng vì mỗi xe đáng 2000 nên cost TB trên 12 instance không khác biệt có ý nghĩa (−0,17%, Wilcoxon $p = 0{,}64$).

---

## Ablation

* 12 instance 100 khách, 2 cho mỗi nhóm của bộ Wang & Chen: Cdp1 (cdp101, cdp103), Cdp2 (cdp203, cdp207), Rdp1 (rdp103, rdp107), Rdp2 (rdp201, rdp210), RCdp1 (rcdp101, rcdp104), RCdp2 (rcdp202, rcdp207). Instance nhỏ hơn không dùng vì mọi cấu hình đều đạt cùng nghiệm.
* 30 run, seed 42–71, `pop_size = 30`, `max_iter = 1000`, Kaggle T4. Mỗi cột chỉ tắt một cải tiến so với fullcaitienGPU (−mục 2: `--init sa`; −mục 3: `--ls_scope best --ls_strategy first --no_ls_2opt_star`). Hai cột này chưa đưa mọi phần về đúng như baseline:
  * −mục 2: route elimination định kỳ mỗi 25 thế hệ vẫn bật (CLI không có flag để tắt).
  * −mục 3: LS định kỳ chỉ còn cho cá thể tốt nhất, first improvement, không 2-opt\*; nhưng lúc khởi tạo LS first improvement (không 2-opt\*) vẫn chạy một lần cho cả quần thể, sau Route Elimination và trước Ejection Pool.
* Thống kê (chốt cho cả lần so sánh 56 instance): từng instance dùng Mann–Whitney hai phía trên 30 run, hiệu chỉnh Holm trên 12 instance; cả bộ dùng Wilcoxon hai phía trên 12 cặp cost TB; độ ổn định dùng Levene (trung vị), hiệu chỉnh Holm.

| Nhóm | Instance | fullcaitienGPU | −mục 2 | −mục 3 |
|---|---|---|---|---|
| Cdp1 | cdp101 | 23001.1 (11) | 23204.4 (11.1) | 23052.2 (11) + |
| Cdp1 | cdp103 | 20965.6 (10) | 21265.0 (10.1) | 21078.1 (10) + |
| Cdp2 | cdp203 | 6593.6 (3) | 6593.6 (3) | 6608.8 (3) + |
| Cdp2 | cdp207 | 6588.3 (3) | 6588.3 (3) | 6588.3 (3) |
| Rdp1 | rdp103 | 28728.3 (13.7) | 27346.9 (13) − | 27773.9 (13.2) |
| Rdp1 | rdp107 | 22978.9 (10.9) | 23117.6 (11) | 22254.3 (10.5) |
| Rdp2 | rdp201 | 9260.8 (4) | 9263.8 (4) | 9353.9 (4) + |
| Rdp2 | rdp210 | 6967.6 (3) | 6974.1 (3) | 7083.1 (3) + |
| RCdp1 | rcdp101 | 31481.3 (14.9) | 33703.1 (16) + | 31527.9 (14.9) + |
| RCdp1 | rcdp104 | 21237.0 (10) | 23177.3 (11) + | 21324.5 (10) + |
| RCdp2 | rcdp202 | 7926.0 (3.23) | 9166.8 (4) + | 7824.9 (3.13) |
| RCdp2 | rcdp207 | 7134.6 (3) | 7719.7 (3.3) + | 7240.3 (3) + |
| | **ΔCost TB so với fullcaitienGPU** | 0 | +3,19% | −0,17% |
| | **fullcaitienGPU tốt hơn / ngang / kém hơn** | — | 4 / 7 / 1 | 8 / 4 / 0 |
| | **$p$ (Wilcoxon, 12 instance)** | — | 0,042 | 0,638 |
| | **sd nhỏ hơn / ngang / lớn hơn** | — | 1 / 8 / 3 | 5 / 7 / 0 |
| | **Thời gian TB (s/run)** | 53 | 10 | 55 |

* Mỗi ô: cost TB của 30 run (NV TB); `+` / `−` khi fullcaitienGPU tốt hơn / kém hơn có ý nghĩa ở instance đó.
* ΔCost TB: trung bình trên 12 instance của (cost TB cột − cost TB fullcaitienGPU) / cost TB fullcaitienGPU.

**Tổng hợp**:

| Cải tiến | Kết quả |
|---|---|
| 1. Toàn bộ trên GPU, mỗi cá thể một block | Song song hóa không đổi nghiệm. Khởi tạo như paper (`--init sa`, vẫn bật mục 3): nhanh hơn baseline 23 lần. fullcaitienGPU: nhanh hơn baseline 4,2 lần, cost thấp hơn ở 11/12 instance |
| 2. RCRS-GRASP + Ejection Pool | Cost −3,19% ($p = 0{,}042$), tốt hơn ở 4/12, kém ở 1 (rdp103); lợi từ việc bớt xe ở nhóm RC; thời gian ×5,5 |
| 3. LS cho cả quần thể + Inter 2-opt\* + best improvement song song | TD thấp hơn khi cùng số xe (7/8 instance), ổn định hơn (5/12), cùng thời gian; cost TB không khác biệt có ý nghĩa ($p = 0{,}64$) vì tắt mục 3 đôi khi bớt thêm xe |

Kết quả: `kaggle/runs_ablation/results/` (chạy: `kaggle/orchestrate_ablation.py`; bảng: `kaggle/analyze_ablation.py`). Vài cấu hình bị chia thành nhiều tiến trình (chunk); `analyze_ablation.py` bỏ run đầu của các chunk sau run 1 khi tính thời gian, để JIT chỉ được tính một lần như khi chạy một tiến trình.

Chạy toàn bộ instance (fullcaitienGPU, 30 run mỗi instance trong một tiến trình, cách đo thời gian như trên): `kaggle/orchestrate_gpu_all.py`, kết quả ghi vào `docs/ph_showoa_caitiengpu_all.csv` và `docs/instances_benchmark_pop_size30_runs=30_iter1000.csv`.
