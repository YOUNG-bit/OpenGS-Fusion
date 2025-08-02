# MIT License
#
# # Copyright (c) 2022 Ignacio Vizzo, Cyrill Stachniss, University of Bonn
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


# 从环境变量获取基础路径
if(NOT DEFINED ENV{OPENGS_ENV})
    message(FATAL_ERROR "OPENGS_ENV environment variable is not set. Please set it to your conda environment path (e.g., ～/anaconda3/envs/vdbfusion)")
endif()
set(ENV_PATH $ENV{OPENGS_ENV})

# 设置路径变量
set(CMAKE_PREFIX_PATH "${ENV_PATH}/lib/python3.9/site-packages/torch/share/cmake")
set(CMAKE_LIBRARY_PATH "/usr/local/cuda-11.8/lib64;${ENV_PATH}/lib")

find_package(pybind11 REQUIRED)

# 设置libtorch的路径
set(Torch_DIR "${ENV_PATH}/lib/python3.9/site-packages/torch/share/cmake/Torch/")
find_package(Torch REQUIRED)

# 打印找到的libtorch的路径
message(STATUS "Torch_DIR: ${Torch_DIR}")

# 添加定义以使用旧的CXX11 ABI
# add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)

# 添加定义以解决问题
add_definitions(-DSTRIP_ERROR_MESSAGES)

pybind11_add_module(vdbfusion_pybind vdbfusion_pybind.cpp)
target_compile_options(vdbfusion_pybind PRIVATE -Werror -Wall -Wextra)
target_link_libraries(vdbfusion_pybind PRIVATE VDBFusion::vdbfusion)
# target_link_libraries(vdbfusion_pybind PRIVATE ${TORCH_LIBRARIES})

# PYOPENVDB_SUPPORT is defined only by the existence of the pyopenvdb library.
find_package(Python COMPONENTS Interpreter)
execute_process(COMMAND
                ${PYTHON_EXECUTABLE} "-c" "import pyopenvdb; print(True)"
                OUTPUT_VARIABLE PYOPENVDB_SUPPORT
                ERROR_QUIET)
if(PYOPENVDB_SUPPORT)
  find_package(Boost COMPONENTS python REQUIRED)
  target_compile_definitions(vdbfusion_pybind PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:PYOPENVDB_SUPPORT>")
  target_link_libraries(vdbfusion_pybind PRIVATE Boost::python)
  message(STATUS "PYOPENVDB_SUPPORT enabled")
endif()