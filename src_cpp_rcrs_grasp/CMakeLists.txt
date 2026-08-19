cmake_minimum_required(VERSION 3.18)
project(PH_SHOWOA_CPP LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(ENABLE_CUDA "Enable CUDA support" OFF)

find_package(OpenMP REQUIRED)

set(SOURCES
    main.cpp
    data.cpp
    compute_backend.cpp
    operator.cpp
    search_framework.cpp
)

if(ENABLE_CUDA)
    if(WIN32)
        find_program(CUDA_HOST_COMPILER NAMES cl.exe)
        if(NOT CUDA_HOST_COMPILER)
            message(FATAL_ERROR "ENABLE_CUDA=ON on Windows requires cl.exe in PATH. Open a Visual Studio Developer Command Prompt or set CMAKE_CUDA_HOST_COMPILER before configuring.")
        endif()
        set(CMAKE_CUDA_HOST_COMPILER "${CUDA_HOST_COMPILER}")
    endif()

    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES native)
    endif()

    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)

    add_compile_definitions(USE_CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    list(APPEND SOURCES compute_backend.cu)
endif()

add_executable(phshowoa_cpp ${SOURCES})
target_link_libraries(phshowoa_cpp PRIVATE OpenMP::OpenMP_CXX)

if(ENABLE_CUDA)
    target_link_libraries(phshowoa_cpp PRIVATE CUDA::cudart)
endif()
