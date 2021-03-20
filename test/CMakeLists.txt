cmake_minimum_required(VERSION 3.18)

find_package(CMakeshift REQUIRED)
find_package(Boost REQUIRED COMPONENTS unit_test_framework filesystem)
find_package(Eigen3 REQUIRED)
find_package(matrixmarket REQUIRED)

add_executable(
  test-thrustshift
  "main.cpp"
  "test-coo.cu"
  "test-csr.cu"
  "test-math.cu"
  "test-not-a-vector.cu"
)

include(CMakeshift/TargetCompileSettings)
cmakeshift_target_compile_settings(
  test-thrustshift
  PRIVATE
  default
  diagnostics
  #  $<$<CONFIG:Debug>:runtime-checks>
)

target_compile_features(test-thrustshift PRIVATE cuda_std_17)

target_compile_definitions(test-thrustshift PRIVATE
                           gsl_CONFIG_CONTRACT_VIOLATION_TERMINATES)

if(CMAKE_CUDA_COMPILER_ID MATCHES "NVIDIA")
  # Boost.Preprocessor doesn't trust NVCC to handle variadic macros correctly,
  # cf. https://github.com/boostorg/preprocessor/issues/15
  target_compile_definitions(test-thrustshift PRIVATE
                             $<$<COMPILE_LANGUAGE:CUDA>:BOOST_PP_VARIADICS=1>)
endif()

target_link_libraries(
  test-thrustshift
  PRIVATE Boost::unit_test_framework Boost::filesystem thrustshift Eigen3::Eigen matrixmarket::matrixmarket
)

target_compile_definitions(test-thrustshift PRIVATE EIGEN_NO_CUDA)
target_compile_definitions(test-thrustshift PRIVATE gsl_CONFIG_CONTRACT_CHECKING_AUDIT)

add_test(NAME "test-thrustshift" COMMAND test-thrustshift)
