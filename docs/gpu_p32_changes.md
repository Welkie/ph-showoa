# Flow GPU P32: thay đổi đã áp dụng

Flow chính giữ SA_RCRS_GRASP trên Numba CUDA Graph. `src_python` không bị sửa. Population GPU mặc định 32, kể cả khi bật paper_flags; notebook truyền rõ 32.

## Kế thừa base

- Cost/best selection = 2000*NV+TD; lịch SHO/WOA và công thức nhiệt SA không đổi.
- Sửa main-loop SA GPU để chấp nhận luôn deltaCost<=0.001 như base, gồm cả Python helper dự phòng. Tests dùng base làm oracle quanh ngưỡng này.
- Notebook dùng 100 lượt SA mỗi mức nhiệt (trước đây ghi đè thành 25); T0=100, alpha=.95, Tmin=.1 giữ nguyên.
- Giữ feasibility/coverage/repair hiện tại. Mọi move mới được kiểm tra capacity/time window và chỉ nhận khi scalar cost giảm.

## Các cải tiến nhỏ trong bản này

1. **Inter-route 2-opt***: sau khi hết cải thiện bằng các neighborhood base, thử đổi hai đuôi không rỗng của hai tuyến; giữ NV, kiểm tra hai tuyến rồi mới ghi nhận. Nếu cải thiện, quay lại đầu local search. Dùng buffer sẵn có, không cấp phát hoặc sync host trong search. `--gpu_2opt_star 0` tắt để ablation; mặc định 1.
2. **Best mỗi island**: route elimination và local search chạy trên từng island-best, sau đó cập nhật global-best. Chạy song song trên device; CPU reference có cùng lịch. `--gpu_ls_scope global` khôi phục phạm vi run-best; mặc định `island`. Đây là cải tiến về phạm vi search, không gọi là nguyên bản base.
3. **P32/I4**: 4 islands, mỗi island 8 cá thể, giữ ring migration. Notebook khai báo rõ islands=4. Các cấu hình population/islands không chia hết vẫn có fallback cũ nhưng in thông báo rõ, không còn im lặng.
4. Giữ route elimination với scalar improvement guard, RCRS-GRASP và CUDA Graph hiện có. Capacity proxy và uniform sampling hiện có **chưa được thay đổi**; chúng vẫn là các biến thể khác base đã ghi trong audit, chưa có kết luận rằng tốt hơn criterion/sampling base.

Không thêm full-population local search hoặc đổi nhiệt/cosine schedule theo commit cũ. Cấu hình islands=4/local-search-island là lựa chọn để đánh giá chất lượng, không phải tuyên bố đã tối ưu trên đủ 15 datasets.

## Bám bảng kết quả người dùng

`gpu_d88da6e_targets.json` lưu 15 mốc GPU trong ảnh, gồm NV/TD/AvgRun và population cũ 36. Không lấy cột CPU làm mục tiêu thay thế. Ảnh không đủ xác định NV/TD là best hay mean qua bao nhiêu runs, hoặc biên đo thời gian; do đó đây là tham chiếu lịch sử, không phải tiêu chí tương đương thống kê đã xác minh.

Notebook xuất file riêng `summary_gpu_p32_i4_2optstar.csv` và log riêng, để resume không đọc kết quả cấu hình trước. Sau khi chạy notebook:

```bash
python scripts/compare_gpu_reference.py /kaggle/working/summary_gpu_p32_i4_2optstar.csv --output /kaggle/working/comparison_d88da6e.json
```

Báo cáo có từng instance: NV/TD mới, deltaNV, deltaTD, delta scalar cost, cùng NV hay không, thiếu/failed. TD thấp hơn nhưng tăng NV không bị gắn nhãn tốt hơn nếu scalar cost tăng. Công cụ chỉ so summary; CPU validation trong solver vẫn là bước kiểm tra tuyến. Nó không suy đoán cấu hình chạy từ nội dung CSV hoặc chứng nhận tính hợp lệ của một CSV tùy ý.

## Kiểm chứng thực tế đã chạy

```powershell
.\.venv\Scripts\python.exe -m src_python_gpu_SA_RCRS_GRASP.main --problem dataset/explicit_rcdp1001.vrpsdptw --compute_backend cpu --init sa_rcrs_grasp --paper_flags --pop_size 32 --num_islands 4 --sa_iterations 100 --runs 1 --max_iter 100 --output_per_gens 100
```

RCdp1001: NV=3, TD=348.982364, cost=6348.982364; kết thúc thành công, qua kiểm tra nghiệm CPU. Khớp mốc ảnh 3/348.98 khi làm tròn.

Các tuyến:

- 0 → 6 → 5 → 9 → 10 → 0
- 0 → 4 → 7 → 2 → 0
- 0 → 1 → 3 → 8 → 0

Đây là một run 100 thế hệ của CPU reference dùng cùng logic kernel; không phải kết quả benchmark GPU 30 runs/1000 generations. 53 giây solver báo gồm CPU JIT, không so với 1.7 giây GPU trong ảnh. Máy kiểm tra không có CUDA khả dụng; CUDA simulator kiểm tra logic/replay nhưng không thay thế kiểm tra Driver/JIT trên GPU thật.

Test bổ sung kiểm tra 2-opt* thoát local optimum base (cùng NV và coverage, scalar metadata khớp tuyến), SA sát PRECISION, so benchmark khi NV khác nhau, và graph replay cả hai phạm vi global/island.

Kết quả: nhóm CPU 56 passed, 2 skipped (Torch không có và CUDA thực không khả dụng); nhóm CUDA simulator 16 passed. Notebook JSON hợp lệ và các cell Python thuần parse được. Các kiểm tra này không xác nhận chất lượng trên 14 instance còn lại.
