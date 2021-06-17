#pragma once

#include <gsl-lite/gsl-lite.hpp>

#include <boost/test/data/test_case.hpp>
#include <boost/test/unit_test.hpp>

#include <cuda/runtime_api.hpp>

#include <thrustshift/managed-vector.h>
#include <thrustshift/copy.h>

namespace thrustshift {

template<class MemoryResource>
void touch_all_memory_resource_pages(MemoryResource& memory_resource) {
	auto device = cuda::device::current::get();
	auto stream = device.default_stream();

	for (const auto [k, v] : memory_resource.get_book()) {
		for (const auto& page : v) {
			BOOST_TEST(!page.allocated);
			using T = std::byte;
			const size_t N = k.bytes / sizeof(T);
			managed_vector<T> dst(N);
			gsl_lite::span<T> src(
			    reinterpret_cast<T*>(page.ptr), N);
			async::copy(stream, src, dst);
			stream.synchronize();
			for (size_t i = 0; i < N; ++i) {
				dst[i] = src[i];
			}
		}
	}
}

}
