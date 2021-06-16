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
			managed_vector<std::byte> dst(k.bytes);
			gsl_lite::span<std::byte> src(
			    reinterpret_cast<std::byte*>(page.ptr), k.bytes);
			async::copy(stream, src, dst);
			stream.synchronize();
			for (size_t i = 0; i < k.bytes; ++i) {
				dst[i] = src[i];
			}
		}
	}
}

}
