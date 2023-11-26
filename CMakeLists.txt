cmake_minimum_required(VERSION 3.18)

project(thrustshift VERSION 0.1.0 LANGUAGES CXX CUDA)

option(BUILD_TESTS "Build tests" OFF)
option(BUILD_APPS "Build apps" OFF)

add_subdirectory(src)
if(BUILD_TESTS)
  enable_testing()
  add_subdirectory(test)
endif()

if (BUILD_APPS)
	add_subdirectory(app)
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
  "CUB"
  "sysmakeshift"
  "gsl-lite"
)
