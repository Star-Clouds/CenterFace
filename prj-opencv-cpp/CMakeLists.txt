#1.cmake verson
cmake_minimum_required(VERSION 2.8)


#2.project name
project(demo C CXX)

#opencv4.1.0
find_package(OpenCV REQUIRED)

#3.set environment variable
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

#4.include
include_directories(${CMAKE_CURRENT_LIST_DIR}/cpp)

#5.source directory
file(GLOB TEST_SRC ${CMAKE_CURRENT_LIST_DIR}/cpp/*.cpp
                   ${CMAKE_CURRENT_LIST_DIR}/demo.cpp)
set(TEST_COMPILE_CODE ${TEST_SRC})


add_executable(demo ${TEST_COMPILE_CODE})

target_link_libraries(demo 
                      ${OpenCV_LIBS}
					  )