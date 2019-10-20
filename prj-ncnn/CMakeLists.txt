#1.cmake verson
cmake_minimum_required(VERSION 2.8)

set DIR=your_ncnnlib_path

#2.project name
project(demo C CXX)

#opencv
find_package(OpenCV REQUIRED)

#3.set environment variable
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

#4. include
include_directories(${DIR}/include/ncnn
                    ${CMAKE_CURRENT_LIST_DIR}/cpp)

# ncnn lib
add_library( libncnn STATIC IMPORTED )
set_target_properties(
        libncnn
        PROPERTIES IMPORTED_LOCATION
        ${DIR}/lib/libncnn.a
)

#6.source directory
file(GLOB TEST_SRC ${CMAKE_CURRENT_LIST_DIR}/cpp/*.cpp
                   ${CMAKE_CURRENT_LIST_DIR}/demo.cpp)
set(TEST_COMPILE_CODE ${TEST_SRC})


add_executable(demo ${TEST_COMPILE_CODE})

target_link_libraries(demo 
                      libncnn
                      ${OpenCV_LIBS}
					  )