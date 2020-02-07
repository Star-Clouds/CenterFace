#1.cmake verson
cmake_minimum_required(VERSION 3.1.0)

set (DIR your_mnn_path)

#2.project name
project(demo C CXX)

#opencv
find_package(OpenCV REQUIRED)

#3.set environment variable
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

#4. include
include_directories(${DIR}/include
                    ${CMAKE_CURRENT_LIST_DIR}/cpp)

# mnn lib
set (LIBS
    ${DIR}/build/libMNN.so
    ${OpenCV_LIBS}
)

# add OPENMP support 
FIND_PACKAGE( OpenMP REQUIRED)
if(OPENMP_FOUND)
message("OPENMP FOUND")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

#6.source directory
file(GLOB TEST_SRC ${CMAKE_CURRENT_LIST_DIR}/cpp/*.cpp
                   ${CMAKE_CURRENT_LIST_DIR}/demo.cpp)
set(TEST_COMPILE_CODE ${TEST_SRC})

add_executable(demo ${TEST_COMPILE_CODE})
target_link_libraries(demo ${LIBS})
