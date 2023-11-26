#pragma once

#include <gsl-lite/gsl-lite.hpp>

#include <boost/test/data/test_case.hpp>
#include <boost/test/unit_test.hpp>

#include <thrustshift/copy.h>
#include <thrustshift/defines.h>
#include <thrustshift/managed-vector.h>

namespace thrustshift {

template <class MemoryResource>
void touch_all_memory_resource_pages(MemoryResource& memory_resource) {

	cudaStream_t stream = 0;

	for (const auto& [k, v] : memory_resource.get_book()) {
		for (const auto& page : v) {
			BOOST_TEST(!page.allocated);
			using T = std::byte;
			const size_t N = k.bytes / sizeof(T);
			managed_vector<T> dst(N);
			gsl_lite::span<T> src(reinterpret_cast<T*>(page.ptr), N);
			async::copy(stream, src, gsl_lite::span<T>(dst));
			THRUSTSHIFT_CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
			for (size_t i = 0; i < N; ++i) {
				dst[i] = src[i];
			}
		}
	}
}

} // namespace thrustshift
