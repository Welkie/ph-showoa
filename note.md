- Đây là thuật toán gốc của tác giả nằm ở src_python được trình bày  trong paper nàyjournal.pone.0343262.pdf . Bây giờ tui đang chạy cải tiến code đó thành fullGPU 100% và thay bước SA ở lúc khởi tạo thành SA_RCRS_grasp.src_python_gpu_SA_RCRS_GRASP. Tuy nhiên, code hiện giờ vẫn còn nhiều hạn chế và tối ưu chưa tốt nên kết quả khá tệ, về mặt thời gian thì nhanh hơn. 
Tối ưu hóa tensor các điểm chưa tốt hoặc cải tiến các điểm chưa tốt để nâng cao kết quả
về mặt thời gian thì vẫn nhanh hơn nhưng chạy vẫn còn tệ về performance. Phải ưu tiên NV trước rồi sau đó đến TD. lẽ ra phải tốt về NV và TD như bản
src_cpp_hybrid_SA_RCRS_GRASP 
Phải luôn đảm bảo 100% chạy trên GPU

