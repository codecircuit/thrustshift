cmake_minimum_required(VERSION 3.18)

find_package(CMakeshift REQUIRED)
find_package(cuda-api-wrappers REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(Boost REQUIRED COMPONENTS program_options filesystem)

option(BUILD_APPS_WITH_FAST_MATH
       "use --fast_math or -ffast-math to compile apps (nvcc/clang)" ON
)
option(INSTALL_APPS
  "install apps" OFF
)

################################################################################

include(CMakeshift/TargetCompileSettings)
function(configure_thrustshift_app_target targetname)

  if(CMAKE_CUDA_COMPILER_ID MATCHES "NVIDIA")
    target_compile_definitions(
      ${targetname} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:BOOST_PP_VARIADICS=1>
    )
    if(BUILD_APPS_WITH_FAST_MATH)
      target_compile_options(
        ${targetname} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
      )
    endif()
  endif()

  if(CMAKE_CUDA_COMPILER_ID MATCHES "Clang")
    set_target_properties(${targetname} PROPERTIES CUDA_EXTENSIONS OFF)
    if(BUILD_APPS_WITH_FAST_MATH)
      target_compile_options(
        ${targetname} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-ffast-math>
      )
    endif()

  endif()

  cmakeshift_target_compile_settings(
    ${targetname} PRIVATE default diagnostics-paranoid
  )

  target_compile_features(${targetname} PRIVATE cuda_std_17)
  target_compile_features(${targetname} PRIVATE cxx_std_17)
  target_compile_definitions(
    ${targetname} PRIVATE gsl_CONFIG_CONTRACT_VIOLATION_TERMINATES
  )
  target_compile_definitions(
    ${targetname} PRIVATE $<$<CONFIG:Debug>:gsl_CONFIG_CONTRACT_CHECKING_AUDIT>
  )
  target_compile_definitions(
    ${targetname} PRIVATE $<$<CONFIG:Release>:gsl_CONFIG_DEVICE_CONTRACT_CHECKING_OFF>
  )

endfunction()

################################################################################

add_executable(bench-k-selection "bench-k-selection.cu")

configure_thrustshift_app_target(bench-k-selection)

#######################################################
# FOR RELOCATABLE DEVICE CODE AND DYNAMIC PARALELLISM #
#######################################################
set_target_properties(bench-k-selection PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(
  bench-k-selection
  PRIVATE Boost::program_options thrustshift
          cuda-api-wrappers::nvtx
          CUDA::nvToolsExt # because the cuda-api-wrappers do not export the dependency correctly
)

if(INSTALL_APPS)
  install(TARGETS bench-k-selection RUNTIME)
endif()

################################################################################

add_executable(bench-bin-values256 "bench-bin-values256.cu")

configure_thrustshift_app_target(bench-bin-values256)

#######################################################
# FOR RELOCATABLE DEVICE CODE AND DYNAMIC PARALELLISM #
#######################################################
set_target_properties(bench-bin-values256 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(
  bench-bin-values256
  PRIVATE Boost::program_options thrustshift
          cuda-api-wrappers::nvtx
          CUDA::nvToolsExt # because the cuda-api-wrappers do not export the dependency correctly
)

if(INSTALL_APPS)
  install(TARGETS bench-bin-values256 RUNTIME)
endif()
################################################################################
