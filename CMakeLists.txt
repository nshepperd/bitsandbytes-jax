cmake_minimum_required(VERSION 3.27)
project(bnb_jax LANGUAGES CXX)

set(CMAKE_INSTALL_RPATH "$ORIGIN/nvidia/cuda_runtime/lib")
# Make sure RPATH is used instead of RUNPATH
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH FALSE)

if(EXISTS ${Python_ROOT_DIR}/bin/python)
    execute_process(
        COMMAND ${Python_ROOT_DIR}/bin/python -c "import sysconfig; print(sysconfig.get_path('purelib'))"
        OUTPUT_VARIABLE PYTHON_SITE_PACKAGES
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    message(STATUS "Python site-packages: ${PYTHON_SITE_PACKAGES}")
    set(CMAKE_PREFIX_PATH ${PYTHON_SITE_PACKAGES} CACHE PATH "Path to Python site-packages")
endif()

# Find required packages
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)

if(Python_FOUND)
    message(STATUS "Found Python: ${Python_EXECUTABLE} (found version ${Python_VERSION})")
else()
    message(FATAL_ERROR "Python not found")
endif()

execute_process(
    COMMAND ${Python_EXECUTABLE} -m nanobind --cmake_dir
    OUTPUT_VARIABLE nanobind_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

find_package(nanobind CONFIG REQUIRED)
find_package(CUDAToolkit REQUIRED)

# Create nanobind module
nanobind_add_module(_core
    csrc/bnb_jax/module.cpp
    csrc/bnb_jax/ffi_handlers.cpp
)

target_include_directories(_core PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc
)

target_link_libraries(_core PRIVATE
    CUDA::cudart
    ${CMAKE_DL_LIBS}
)

target_compile_features(_core PRIVATE cxx_std_17)

install(TARGETS _core LIBRARY DESTINATION bnb_jax_lib)
