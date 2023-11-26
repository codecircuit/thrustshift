#pragma once

#include <exception>
#include <iostream>

#define THRUSTSHIFT_FHD __forceinline__ __host__ __device__
#define THRUSTSHIFT_HD __host__ __device__
#define THRUSTSHIFT_FD __forceinline__ __device__
#define THRUSTSHIFT_FH __forceinline__ __host__
#define THRUSTSHIFT_HD __host__ __device__
#define THRUSTSHIFT_H __host__
#define THRUSTSHIFT_D __device__

#define THRUSTSHIFT_CHECK_CUDA_ERROR(err)                            \
	if (err != cudaSuccess) {                                        \
		std::cout << "CUDA error in " << __FILE__ << ":" << __LINE__ \
		          << ", error = " << err << std::endl;               \
		std::terminate();                                            \
	}
