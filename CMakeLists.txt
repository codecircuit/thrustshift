cmake_minimum_required(VERSION 3.18)

project(thrustshift VERSION 0.0.1 LANGUAGES CXX CUDA)

option(BUILD_TESTS "Build tests" OFF)

add_subdirectory(src)
if(BUILD_TESTS)
  enable_testing()
  add_subdirectory(test)
endif()

find_package(CMakeshift 3.7 REQUIRED)
include(CMakeshift/InstallBasicPackageFiles)
cmakeshift_install_basic_package_files(
  thrustshift
  COMPATIBILITY
  SameMajorVersion
  ARCH_INDEPENDENT
  EXPORT
  thrustshift
  DEPENDENCIES
  "CMakeshift"
  "cuda-api-wrappers"
  "CUB"
  "sysmakeshift"
  "gsl-lite"
)
