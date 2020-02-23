### The C++ version of CenterFace with MNN

## requirement
- mnn
- opencv(4.1.1)

## usage 
## 1.compile mnn lib
## 2.set mnn directory in line 4 in the CMakeLists.txt
## for example:
```
set (DIR /home/mirror/workspace/mnn)
```
## 3.compile and run the project
 * cd CenterFace/prj-mnn
 * mkdir build && cd build && cmake .. && make -j3
 * ./demo ../../models/mnn  your_image_path