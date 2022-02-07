cmake_minimum_required(VERSION 3.18)

find_package(CMakeshift REQUIRED)
find_package(Boost REQUIRED COMPONENTS unit_test_framework filesystem)
find_package(Eigen3 REQUIRED)
find_package(matrixmarket REQUIRED)

option(INSTALL_TESTS
  "install tests" OFF
)

add_executable(
  test-thrustshift
  "main.cpp"
  "test-coo.cu"
  "test-csr.cu"
  "test-math.cu"
  "test-not-a-vector.cu"
  "test-sort.cu"
  "test-wrap-subgroups.cu"
  "test-bit-successive-permutation.cu"
  "test-count-leading-zeros.cu"
  "test-block-copy.cu"
  "test-container-conversion.cu"
  "test-memory-resource.cu"
  "test-memory-resource.cpp"
  "test-multi-iterator.cu"
  "test-array-tuple-transform.cu"
  "test-type-traits.cu"
  "test-for-each.cu"
  "test-k-selection.cu"
  "test-tuple-to-string.cu"
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
target_compile_features(test-thrustshift PRIVATE cxx_std_17)

target_compile_definitions(test-thrustshift PRIVATE
                           gsl_CONFIG_CONTRACT_VIOLATION_TERMINATES)

if(CMAKE_CUDA_COMPILER_ID MATCHES "NVIDIA")
  # Boost.Preprocessor doesn't trust NVCC to handle variadic macros correctly,
  # cf. https://github.com/boostorg/preprocessor/issues/15
  target_compile_definitions(test-thrustshift PRIVATE
                             $<$<COMPILE_LANGUAGE:CUDA>:BOOST_PP_VARIADICS=1>)
endif()

#######################################################
# FOR RELOCATABLE DEVICE CODE AND DYNAMIC PARALELLISM #
#######################################################
set_target_properties(test-thrustshift PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(
  test-thrustshift
  PRIVATE Boost::unit_test_framework Boost::filesystem thrustshift Eigen3::Eigen matrixmarket::matrixmarket
)

target_compile_definitions(test-thrustshift PRIVATE gsl_CONFIG_CONTRACT_CHECKING_AUDIT)

add_test(NAME "test-thrustshift" COMMAND test-thrustshift)

if(INSTALL_TESTS) 
  install(TARGETS test-thrustshift RUNTIME)
endif()
