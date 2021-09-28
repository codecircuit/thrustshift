cmake_minimum_required(VERSION 3.18)

find_package(CMakeshift REQUIRED)
find_package(gsl-lite REQUIRED)
find_package(cuda-api-wrappers REQUIRED)
find_package(CUB REQUIRED)
find_package(sysmakeshift REQUIRED)
find_package(cuda-kat REQUIRED)

add_library(thrustshift INTERFACE)

target_include_directories(
  thrustshift
  INTERFACE "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>"
)
target_compile_features(thrustshift INTERFACE cxx_std_14)

target_link_libraries(
  thrustshift
  INTERFACE
    gsl::gsl-lite-v1 cuda-api-wrappers::runtime-api sysmakeshift::sysmakeshift
    CUB::CUB
    cuda-kat::cuda-kat
)

install(
  DIRECTORY
    "${PROJECT_SOURCE_DIR}/include/" # the trailing '/' is significant
  TYPE INCLUDE
)
install(TARGETS thrustshift EXPORT thrustshift INCLUDES)
