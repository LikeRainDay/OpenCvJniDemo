# Sets the minimum version of CMake required to build the native
# library. You should either keep the default value or only pass a
# value of 3.4.0 or lower.

cmake_minimum_required(VERSION 3.4.1)

set(opencv_root_dir ${CMAKE_SOURCE_DIR}/opencv)
set(INC_DIR ${opencv_root_dir}/include)
set(opencv_lib_dir ${opencv_root_dir}/libs)

#添加头文件目录
include_directories(${INC_DIR})
#添加外部动态链接库
add_library(lib_opencv  SHARED   IMPORTED)
set_target_properties(lib_opencv     PROPERTIES     IMPORTED_LOCATION
     ${opencv_lib_dir}/${ANDROID_ABI}/libopencv_java3.so)

add_library(dynamic-deal

           SHARED

           src/main/cpp/dynamic-deal.cpp
           src/main/cpp/lang.h
           src/main/cpp/dynamic-deal-head.h )

add_library(state-deal

            SHARED

            src/main/cpp/state-deal.cpp
            src/main/cpp/lang.h
            src/main/cpp/state-deal-header.h)


add_library(int-deal

            SHARED

            src/main/cpp/int-deal.cpp
            src/main/cpp/int-deal-head.h)

find_library(log-lib
              # Specifies the name of the NDK library that
              # you want CMake to locate.
              log )

#dynamic-deal

#静态图片处理生成的.so链接的目标Libaries
target_link_libraries( state-deal
                       lib_opencv

                       ${log-lib})
#int图片处理生成的.so链接的目标Libaries
target_link_libraries(int-deal
                      lib_opencv

                      ${log-lib})
#动态图片处理生成的.so链接的目标Libaries
target_link_libraries(dynamic-deal
                      lib_opencv

                      ${log-lib})