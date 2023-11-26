cmake_minimum_required(VERSION 3.18)

find_package(CMakeshift REQUIRED)
find_package(gsl-lite REQUIRED)
find_package(cuda-api-wrappers REQUIRED)
find_package(CUB REQUIRED)
find_package(sysmakeshift REQUIRED)
find_package(makeshift REQUIRED)

add_library(thrustshift INTERFACE)

target_include_directories(
  thrustshift
  INTERFACE "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>"
# "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>" # if installed with spack, this variable expands to nothing
  "$<INSTALL_INTERFACE:include>"
)
target_compile_features(thrustshift INTERFACE cxx_std_17)

# For compatibility with CUDA 12: usage of legacy CUDA Dynamic Parallelism
target_compile_definitions(thrustshift INTERFACE CUDA_FORCE_CDP1_IF_SUPPORTED)

target_link_libraries(
  thrustshift
  INTERFACE
    gsl::gsl-lite-v1 cuda-api-wrappers::runtime-and-driver
    sysmakeshift::sysmakeshift
    CUB::CUB
    makeshift::makeshift
)

install(
  DIRECTORY
    "${PROJECT_SOURCE_DIR}/include/" # the trailing '/' is significant
  TYPE INCLUDE
)
install(TARGETS thrustshift EXPORT thrustshift INCLUDES)
