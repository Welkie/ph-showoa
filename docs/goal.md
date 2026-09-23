Đây là thuật toán gốc của tác giả nằm ở src_python được trình bày trong paper: journal.pone.0343262.pdf . Bây giờ tui đang chạy cải tiến code đó thành fullGPU 100% và thay bước SA ở lúc khởi tạo thành SA_RCRS_grasp: thư mục src_python_gpu_SA_RCRS_GRASP. 

Tác giả paper đã code bằng java. Còn này tui code trên python thì sửa thành full gpu nhưng phải là trên python

Chỉ CPU các phần, các bước bên dưới bưới nào GPU đc thì cứ để GPU:
đọc dataset và upload dữ liệu ban đầu;
tạo seed/config rồi copy lên GPU;
vòng host để launch các CUDA kernels;
copy status từ GPU về CPU;
decode nghiệm cuối;
run_best.check(data, false) sau mỗi run;
best_solution.check(data) ở cuối để hậu kiểm
hoặc các phần phụ không liên quan
ý tui cpu ko được nằm bên trong 1 run: ví dụ: cpu -> run(full gpu) - thuật toán -> cpu. Miễn là toàn bộ cái showoa hoàn toàn trên gpu, còn mấy cái phụ phụ thì ko quan trọng

Flow này:
CPU_PREP run=N
    ↓
RUN_GPU_BEGIN run=N
    ↓
Toàn bộ initialization + SA + RCRS-GRASP +
SHO/WOA + objective + acceptance +
local search + migration trên CUDA
    ↓
RUN_GPU_END run=N
    ↓
CPU_DECODE sau khi hoàn tất toàn bộ runs

đảm bảo đúng flow này cho tui

File code tui chạy trên kaggle: kaggle_gpu_sa_rcrs_grasp

Tham khảo bài báo GPU acceleration hoặc các bài khác liên quan phù hợp

Mục tiêu: Vượt hoặc bằng target ở file ph_showoa_python_target 


Flow:
CPU_PREP — trước các run
│  Đọc dataset, parse config
│  Cấp phát một lần toàn bộ buffer GPU, kể cả RNG
│  Upload dataset và config
│  Dựng graph thực thi, khai báo phụ thuộc giữa các bước
│
├─ Với mỗi run:
│
│  CPU
│  │  Truyền seed / run_id
│  │  Launch graph một lần
│  ▼
│  RUN_GPU_BEGIN
│  │
│  ├─ Khởi tạo trạng thái
│  │    Khởi tạo RNG trên GPU
│  │    Reset ibest, gbest và bộ đếm thế hệ
│  │
│  ├─ Khởi tạo quần thể — giữ thứ tự hiện tại
│  │    RCRS-GRASP
│  │      → SA warmup
│  │      → Route elimination
│  │      → Local search
│  │      → Cập nhật ibest
│  │      → Cập nhật gbest
│  │
│  ├─ Vòng thế hệ trên GPU
│  │    WHILE gen ≤ max_iter:
│  │
│  │      1. Tính a, p_hybrid, p_mode
│  │
│  │      2. SHO/WOA
│  │           Chọn cá thể tham chiếu
│  │           Sinh và đánh giá nghiệm ứng viên
│  │           Kiểm tra khả thi, quyết định chấp nhận
│  │           Chuyển sang quần thể thế hệ mới
│  │
│  │      3. Nếu đến lịch local search:
│  │           Route elimination → Local search
│  │
│  │      4. Nếu đến lịch diversification:
│  │           Đa dạng hóa quần thể
│  │
│  │      5. Nếu đến lịch migration:
│  │           Di cư giữa các đảo
│  │
│  │      6. Cập nhật ibest → cập nhật gbest
│  │
│  │      7. Ghi NV/TD vào buffer log trên GPU
│  │           Tăng gen
│  │
│  ├─ Lưu gbest của run
│  ▼
│  RUN_GPU_END
│
│  CPU
│     Nhận gbest và log của run
│     Decode → check()
│     Cập nhật nghiệm tốt nhất giữa các run đã hậu kiểm
│
└─ CPU: xuất nghiệm tốt nhất → check() cuối