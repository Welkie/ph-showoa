from numba import cuda

# 1. Kiểm tra xem Numba có thể giao tiếp với CUDA (GPU) không
cuda_available = cuda.is_available()
print(f"CUDA có sẵn không?: {cuda_available}")

print("-" * 40)

if cuda_available:
    # 2. Lấy danh sách tất cả các GPU có trong máy
    gpu_list = cuda.gpus
    print(f"Số lượng GPU tìm thấy: {len(gpu_list)}")
    
    print("-" * 40)
    
    # 3. Duyệt qua từng GPU và in ra thông tin chi tiết
    for index, gpu in enumerate(gpu_list):
        # Tên của GPU thường ở dạng bytes, chúng ta cần decode sang chuỗi ký tự (str)
        gpu_name = gpu.name.decode('utf-8') if isinstance(gpu.name, bytes) else gpu.name
        print(f"ID Thiết bị: {index}")
        print(f"  - Tên GPU: {gpu_name}")
        
        # Kiểm tra xem đây có phải là GPU đang được chọn mặc định không
        # (Numba mặc định chọn GPU đầu tiên - ID 0 nếu bạn không chỉ định)
        current_device = cuda.get_current_device()
        if current_device.id == gpu.id:
            print("  - Trạng thái: Đang được kích hoạt mặc định (Current Device)")
            
else:
    print("❌ Không tìm thấy GPU NVIDIA hoặc Driver CUDA chưa được cài đặt/cấu hình đúng.")