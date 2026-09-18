Đây là thuật toán gốc của tác giả nằm ở src_python được trình bày trong paper: journal.pone.0343262.pdf . Bây giờ tui đang chạy cải tiến code đó thành fullGPU 100% và thay bước SA ở lúc khởi tạo thành SA_RCRS_grasp: thư mục src_python_gpu_SA_RCRS_GRASP. Tuy nhiên, code hiện giờ vẫn còn nhiều hạn chế và tối ưu chưa tốt nên kết quả khá tệ, về mặt thời gian thì nhanh hơn. 
Tối ưu hóa tensor các điểm bên trong còn chưa tốt. Bạn tìm cách cải tiến các điểm chưa tốt để nâng cao kết quả, kết quả về mặt thời gian thì vẫn nhanh hơn nhưng chạy vẫn còn tệ về performance. Phải ưu tiên NV trước rồi sau đó đến TD. lẽ ra phải tốt về NV và TD như bản src_cpp_hybrid_SA_RCRS_GRASP (tham khảo ý tưởng, logic, mọi thứ) khác là bản đó chạy hybrid(cpu+gpu), còn tui muốn là full gpu

Phải luôn đảm bảo 100% chạy trên GPU

Tác giả paper đã code bằng java. Còn này tui code trên python thì sửa thành full gpu nhưng phải là trên python

Chỉ CPU các phần:
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