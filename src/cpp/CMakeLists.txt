cmake_minimum_required(VERSION 3.0)

project(cudaSift)


# Find CUDA
find_package(CUDA REQUIRED)

# Find blas/lapack and openmp
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set cuda architecture
set(CUDA_ARCH "89" CACHE STRING "CUDA architecture")


if (WIN32)
    set(EXTRA_CXX_FLAGS "/DVERBOSE /D_CRT_SECURE_NO_WARNINGS ")
    list(APPEND CUDA_NVCC_FLAGS "-arch=${CUDA_ARCH};--compiler-options;-O2;-DVERBOSE") 
endif()
if (UNIX)
    if (APPLE)
        set(EXTRA_CXX_FLAGS "-DVERBOSE -msse2")
        list(APPEND CUDA_NVCC_FLAGS "-arch=${CUDA_ARCH};--compiler-options;-O2;-DVERBOSE") 
    else()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -msse2 ")
        list(APPEND CUDA_NVCC_FLAGS "-lineinfo;--compiler-options;-O2;-D_FORCE_INLINES;-DVERBOSE_NOT") 
    endif()
endif()

# append fast math flag for cuda
list(APPEND CUDA_NVCC_FLAGS "--use_fast_math")

# All warnings, and treat as errors
if (WIN32)
    set(EXTRA_CXX_FLAGS "${EXTRA_CXX_FLAGS} /W4 /WX")
else()
    set(EXTRA_CXX_FLAGS "${EXTRA_CXX_FLAGS} -Wall -Wextra -Werror -Wpedantic")
endif()

set(cuda_sources
    cudaImage.cu
    cudaImage.h
    cudaSiftH.cu
    cudaSiftH.h
    matching.cu
    cudaSiftD.h
    cudaSift.h
    cudautils.h
)

set(sources
    cuSIFT.cpp
    geomFuncs_blas.cpp
)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CUDA_INCLUDE_DIRS}
)

# compile shared library, entry point is cuSIFT.cpp
cuda_add_library(cuSIFT SHARED ${cuda_sources} ${sources} OPTIONS -arch=sm_${CUDA_ARCH})
target_link_libraries(cuSIFT ${CUDA_cudadevrt_LIBRARY} lapacke openblas)

install(TARGETS cuSIFT DESTINATION lib)
install(FILES cuSIFT.hpp DESTINATION include)


#imwback cuda library
cuda_add_library(imwback SHARED imwback.cu)
target_link_libraries(imwback ${CUDA_cudadevrt_LIBRARY})

install(TARGETS imwback DESTINATION lib)
install(FILES imwback.h DESTINATION include)



# Test program, check if we have opencv
find_package(OpenCV)

if (NOT OpenCV_FOUND)
    message(WARNING "OpenCV not found, not building test program")
    return()
endif()

add_executable(test_cuSIFT test.cpp ${sources} ${cuda_sources})
target_include_directories(test_cuSIFT PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(test_cuSIFT cuSIFT imwback ${CUDA_LIBRARIES} ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES} ${OpenCV_LIBS})
install(TARGETS test_cuSIFT DESTINATION bin)

