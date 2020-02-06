### the C++ version of centerface with mnn by MirrorYuChen

## requirement
- mnn
- opencv(4.1.1)

## usage 
## 1.compile mnn lib
## you can refer to csdn blog: https://blog.csdn.net/sinat_31425585/article/details/101606437
## 2.set mnn directory in line 4 in the CMakeLists.txt
## for example:
```
set (DIR /home/mirror/workspace/mnn)
```
## 3.compile and run the project
 * cd centerface/prj-mnn
 * mkdir build && cd build && cmake .. && make -j3
 * ./demo ../../models/mnn  your_image_path