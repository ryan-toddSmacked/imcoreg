cmake_minimum_required(VERSION 3.0)

project(imcoreg_matlab)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Matlab REQUIRED)

# find cuda
find_package(CUDA REQUIRED)

include_directories(${Matlab_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS})

matlab_add_mex(NAME imcoregmx SRC imcoregmx.cpp)

set(LIBDIR ${CMAKE_CURRENT_SOURCE_DIR}/lib)
set(INCDIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_link_directories(imcoregmx PUBLIC ${LIBDIR})
target_link_libraries(imcoregmx ${Matlab_LIBRARIES} cuSIFT imwback ${CUDA_LIBRARIES})
target_include_directories(imcoregmx PUBLIC ${INCDIR})


