Các cải tiến của src_python_gpu_SA_RCRS_GRASP so với src_python có thể chia thành hai nhóm.
Cải tiến thuật toán
- RCRS–GRASP initialization
  Khởi tạo nghiệm bằng RCRS kết hợp Restricted Candidate List của GRASP. Tham số alpha được lấy trong khoảng cấu hình để tăng độ đa dạng quần thể.
  [gpu_kernels.py (line 225)](D:/AI_Research/PH-SHOWOA/src_python_gpu_SA_RCRS_GRASP/gpu_kernels.py:225)
- Route elimination
  Thử loại tuyến ngắn nhất rồi chèn lại khách vào các tuyến còn lại. Nghiệm chỉ được giữ nếu scalar cost 2000×NV+TD giảm.
  [gpu_kernels.py (line 421)](D:/AI_Research/PH-SHOWOA/src_python_gpu_SA_RCRS_GRASP/gpu_kernels.py:421)
- Island model
  Chia quần thể thành nhiều island, mỗi island có best riêng và tìm kiếm tương đối độc lập. Điều này giúp duy trì đa dạng và giảm hội tụ sớm.
  [gpu_engine.py (line 115)](D:/AI_Research/PH-SHOWOA/src_python_gpu_SA_RCRS_GRASP/gpu_engine.py:115)
- Ring migration
  Theo chu kỳ, best của island này được chuyển sang island kế tiếp nếu tốt hơn cá thể xấu nhất bên nhận.
  [gpu_kernels.py (line 961)](D:/AI_Research/PH-SHOWOA/src_python_gpu_SA_RCRS_GRASP/gpu_kernels.py:961)
- Diversification theo từng island
  Khi stagnation đạt ngưỡng, một phần non-elite population được ruin-and-recreate, còn elite được bảo vệ. Tỉ lệ diversify có thể cấu hình.
  [gpu_kernels.py (line 794)](D:/AI_Research/PH-SHOWOA/src_python_gpu_SA_RCRS_GRASP/gpu_kernels.py:794)
- RCRS residual-capacity proxy
  Điểm chèn RCRS dùng một proxy capacity nhẹ hơn để phù hợp việc đánh giá song song trên GPU. Đây là xấp xỉ có chủ ý, không phải bản sao nguyên công thức tc của base.
Cải tiến thực thi GPU
- Toàn bộ population nằm trên device
  Route, route length, số tuyến, distance và cost dùng các mảng có kích thước cố định. Không cần tạo nhiều Python Solution trong vòng lặp tìm kiếm.
- CUDA Graph cho toàn bộ một run
  Các thế hệ được dựng thành một graph rồi replay bằng một lần launch cho mỗi run. CPU không điều khiển từng generation.
  [gpu_graph.py (line 66)](D:/AI_Research/PH-SHOWOA/src_python_gpu_SA_RCRS_GRASP/gpu_graph.py:66)
- Không đồng bộ CPU–GPU giữa các generation
  Chỉ copy nghiệm tốt nhất và history về CPU sau khi hoàn thành run. Điều này giảm đáng kể launch và synchronization overhead.
- Cập nhật population song song và đồng bộ
  Mỗi agent được xử lý độc lập, ghi vào next-buffer. Sau một generation, current/next buffer được hoán đổi.
- Ping-pong buffers
  Dùng hai population buffer thay vì cấp phát hoặc copy lại toàn bộ dữ liệu mỗi generation.
- Scratch buffers tái sử dụng
  Route tạm, danh sách khách chưa định tuyến, flags và scores được cấp phát một lần rồi tái sử dụng.
- Device-side scheduling
  GPU tự tính a, pHybrid, lịch local search, stagnation, diversification và migration.
- Device-side RNG
  Mỗi agent có trạng thái Xorshift128 riêng. Seed được reset xác định cho từng run, hỗ trợ tái lập kết quả.
- Best tracking nhiều tầng
  Theo dõi population → island best → global best ngay trên GPU, không cần tải toàn quần thể về CPU.
- Feasibility và repair trên device
  Capacity, time window, coverage, duplicate/missing customer và repair đều được xử lý trong kernel.
  [gpu_base_operators.py (line 125)](D:/AI_Research/PH-SHOWOA/src_python_gpu_SA_RCRS_GRASP/gpu_base_operators.py:125)
- Kích thước route buffer theo số khách
  Không còn trần cứng 105 tuyến; buffer có thể chứa tối đa một tuyến không rỗng cho mỗi khách.
- CPU reference dùng cùng kernel logic
  Có thể chạy cùng thuật toán bằng Numba CPU để debug và đối chiếu mà không cần GPU.
- Torch không còn là dependency bắt buộc
  Đường chạy chính dùng Numba CUDA/CUDA Graph; môi trường không cài Torch vẫn chạy được.
Phần SHO, WOA, SA acceptance, SA initialization, feasibility/repair và combined local search hiện là logic bám base/paper. Chúng không được xem là cải tiến riêng của GPU; cải tiến chủ yếu nằm ở RCRS-GRASP, route elimination, island/diversification và kiến trúc thực thi CUDA Graph.

Bước initialization là cải tiến SA_RCRS_GRASP, theo chuỗi:
RCRS-GRASP construction
→ feasibility/repair
→ Simulated Annealing refinement
→ route elimination
→ initial population
Trong đó:
- RCRS-GRASP tạo nghiệm ban đầu đa dạng.
- SA cải thiện từng nghiệm bằng 5 neighborhood và trả về best-so-far.
- Route elimination là bước tăng cường bổ sung sau SA.