# Kiểm tra TD: d88da6e → 0477571

> Cập nhật phạm vi theo yêu cầu người dùng: xem `gpu_base_alignment_review.md`. `src_python` mới là chuẩn, còn d88da6e chỉ là nguồn tham khảo các cải tiến. Giữ pop_size=32. Cost scalar, thang nhiệt theo cost, lịch SHO/WOA và local search trên run-best là hành vi base cần giữ; các đề xuất phục hồi hành vi cũ dưới đây chỉ có ý nghĩa như thử nghiệm cải tiến riêng, không phải sửa cho đúng base. Khuyến nghị P=36 ở cuối tài liệu này không còn áp dụng.

Đối chiếu commit `d88da6eb892e3ccc3f17d4fdcb024104d0ef516e` đã có trong Git local với HEAD `0477571`. Người dùng xác nhận vấn đề là **TD lớn hơn**. Chưa có instance/log cũ–mới để xác định tỷ trọng nguyên nhân trên benchmark thực tế. Không thay đổi solver trong lần kiểm tra này.

## Kết luận có bằng chứng

### 1. Mất inter-route 2-opt* trong local search

Bản cũ `gpu_kernels.deep_local_search_single` có đổi đuôi hai tuyến (2-opt*). Bản hiện tại `gpu_base_operators.local_search` chỉ có intra-route 2-opt, relocate và inter-route swap. Đảo đoạn trong một tuyến không thay thế được đổi đuôi giữa hai tuyến.

Đã dựng trường hợp 16 khách hàng, capacity 4, delivery 1, pickup 0, cửa sổ thời gian rộng, 4 xe. Chạy local search hiện tại đến điểm dừng, rồi chạy đúng local-search kernel lấy từ commit cũ trên cùng nghiệm:

| Trạng thái | NV | TD |
|---|---:|---:|
| Điểm dừng local search hiện tại | 4 | 772.7120002171 |
| Sau local search cũ, 2 passes | 4 | 743.5255347494 |

Giảm 29.1864654676, tương đương 3.78%. Kết quả được tính lại bằng evaluator hiện tại, hợp lệ và đủ khách hàng. Ngoài kết quả tổng hợp trên, phép đổi đuôi trực tiếp tại cuts (3,3) của tuyến index (1,3) đã giảm 25.5241886953: chứng minh neighborhood bị thiếu thực sự có bước cải thiện tại điểm dừng của bản mới.

Tái hiện từ repo root:

```powershell
.\.venv\Scripts\python.exe scripts/probe_td_neighborhood.py
```

Script dùng seed 20260923, tìm thấy tại case index 5, in tọa độ và các tuyến. Dùng logic Python của kernel với NUMBA_DISABLE_JIT=1; đây là chứng cứ về neighborhood, không phải đo hiệu năng CUDA hay mức suy giảm trên dataset thực tế.

### 2. Notebook vô tình làm mất mô hình nhiều islands

`kaggle_gpu_sa_rcrs_grasp.ipynb`: POP_SIZE đổi 36 → 32; không truyền `--num_islands`. `Data` mặc định 6 islands. `GpuEngine.__init__` đặt islands=1 nếu population không chia hết cho islands.

- Cũ: 36/6 → 6 islands, mỗi island 6 cá thể.
- Mới theo notebook: 32%6 != 0 → 1 island, 32 cá thể.
- Ring migration chỉ chạy khi num_islands > 1, nên bị vô hiệu hóa trong cấu hình mới.

Cơ chế này được xác nhận từ code; tác động định lượng đến TD cần A/B. Cần đối chiếu dòng log P/islands của các lần chạy thực tế. Với P=32 có thể chọn rõ 4 hoặc 8 islands, nhưng khi so sánh hồi quy nên giữ P=36/islands=6 như baseline trước.

### 3. Local search từ toàn quần thể chuyển sang chỉ global best

Bản cũ: khởi tạo mỗi cá thể bằng construction → SA → elimination → local search; định kỳ elimination và local search cho toàn bộ population, tối đa 2 passes.

Bản hiện tại: initialization bỏ local search cuối; định kỳ chỉ chạy trên gbest. `gpu_graph.py` launch eliminate/search với 1 block, 1 thread. Local search mới chạy đến điểm dừng thay vì giới hạn 2 passes, nhưng chỉ khai thác một nghiệm, không cải thiện các nghiệm khác trong population.

Đây là thay đổi chiến lược search, không phải hệ quả bắt buộc của CUDA Graph. Có thể giữ Graph và phục hồi local search song song trên population, hoặc chọn elite mỗi island để cân bằng thời gian. Chưa đo phương án nào tốt nhất trên benchmark.

## Các thay đổi ảnh hưởng quỹ đạo search

### SA chấp nhận bước tăng TD dễ hơn

Main loop cũ: khi cùng NV, xác suất exp(-deltaTD / (epsilon + T*TD)); không chấp nhận tăng NV. Bản mới: exp(-deltaCost / (epsilon + T*abs(cost))), cost=2000*NV+TD; có thể chấp nhận tăng NV.

Ví dụ cùng NV=10, TD=1000, tăng TD=100 tại T=0.5:

- Cũ: exp(-100/500) ≈ 0.81873.
- Mới: exp(-100/10500) ≈ 0.99052.

Nhiệt hiệu dụng theo TD tăng 21 lần trong ví dụ này. Đây là thay đổi có chủ ý để bám base, và tests đang xác nhận công thức mới. Nó có thể làm quần thể khó ổn định TD, nhất là khi bỏ local search toàn quần thể; chưa chứng minh là nguyên nhân chính của benchmark.

Warmup cũng đổi từ exp(-deltaTD/T) sang exp(-deltaTD/(T*TD)), cooling 0.85 → 0.95, Tmin 0.5 → 0.1. Bản cũ gọi hardcoded sa_iters=20 (~660 lượt đề xuất/cá thể). Bản mới tôn trọng sa_iterations; notebook truyền 25 (~3375 lượt), mặc định không truyền là 100 (~13500 lượt). Vì vậy cùng max_iter không có nghĩa cùng lượng tính toán. Bản mới có giữ best warmup; không nên kết luận chỉ từ nhiệt rằng init chắc chắn kém hơn.

### Lịch SHO/WOA và WOA operator đã đổi

Xác suất chọn SHO cũ là 0.5*(1+cos(pi*t/T)); mới là max(0.15, 0.5*(1-t/T)). Đầu run giảm từ 1 xuống 0.5; giữa run giảm 0.5 xuống 0.25. WOA hiện tại relink theo các khách chung giữa hai tuyến cùng index. Khi số khách chung <2, relink không làm gì. Cần đo tỷ lệ offspring thực sự thay đổi trước khi kết luận hiệu quả của WOA mới.

### Hàm mục tiêu, feasibility và construction không còn giống baseline

- Best selection từ lexicographic NV rồi TD thành scalar 2000*NV+TD. `--objective lexicographic` hiện chỉ được chấp nhận như alias, không phục hồi hành vi cũ.
- Bỏ VEHICLES+3; evaluator mới kiểm tra giới hạn xe và đủ/không trùng khách hàng. So sánh phải ghi cả NV và kiểm tra hợp lệ theo cùng quy tắc.
- RCRS scoring đổi từ deltaTD + 0.5*capacity_penalty + 0.3*old_savings_penalty sang deltaTD + lambda1*capacity_penalty - lambda2*roundtrip_savings; không phải thay đổi cơ học về backend.

## Không nên rollback toàn bộ bản cũ

Bản cũ cũng có lỗi liên quan trực tiếp độ tin cậy TD: cuối diversification gọi `eval_solution(...)` nhưng bỏ giá trị trả về, không cập nhật nr/dist/cost sau khi đổi tuyến. Best trackers có thể dùng số đo cũ không còn khớp nghiệm. Evaluator cũ thiếu coverage check. Do đó log Run Best TD cũ cần đối chiếu với TD tính lại từ các tuyến, không thể mặc định mọi con số thấp hơn đều là nghiệm tốt hơn hợp lệ.

Giữ các sửa chữa evaluator/repair/cache điểm số hiện tại, chỉ phục hồi có chọn lọc các neighborhood/chiến lược tốt.

## Kiểm tra đã chạy và giới hạn

- CPU tests: `test_gpu_base_parity.py` + `test_gpu_sa_acceptance.py`: 38 passed, 1 skipped (torch không có). Có warning ghi pytest cache, không ảnh hưởng kết quả test.
- CUDA simulator: `test_gpu_search_graph.py`: 9 passed. Kiểm tra replay/schedule/RNG trên các case nhỏ.
- CUDA thực không khả dụng (`cuda.is_available() == False`). Simulator không xác nhận CUDA Driver Graph API/JIT hoặc tốc độ GPU.
- Tests parity chứng minh bám base và lịch chạy nhất quán; không chứng minh chất lượng tối ưu tương đương commit cũ.

## Thứ tự thử nghiệm đề xuất

1. Đồng nhất population/islands, seed list, instance, runs, objective/feasibility; ghi NV, TD tính lại và thời gian. Khởi đầu P=36/islands=6 như baseline.
2. Thêm lại 2-opt* như một tùy chọn GPU extension; đây là điểm có bằng chứng thực nghiệm trực tiếp.
3. A/B phạm vi local search: global best, best mỗi island, toàn population; giữ CUDA Graph.
4. Sau đó mới A/B lịch SHO/WOA và thang nhiệt SA riêng từng yếu tố. Nếu vẫn yêu cầu đúng công thức base thì dùng chế độ mở rộng riêng, không âm thầm thay công thức.
5. Báo cáo cả ngân sách số thế hệ và thời gian tương đương; so NV trước khi diễn giải chênh lệch TD.
