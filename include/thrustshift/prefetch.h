#pragma once

#include <type_traits>

#include <cuda/runtime_api.hpp>

namespace thrustshift {

namespace async {

template <class Range, class Device>
void prefetch(cuda::stream_t& stream, Device&& device, Range&& r) {
	using T_ = typename std::remove_reference<Range>::type::value_type;
	// Due to missing const_range in cuda-api-wrappers
	using T = typename std::remove_const<T_>::type;
	cuda::memory::managed::region_t region{const_cast<T*>(r.data()), r.size() * sizeof(T)};
	cuda::memory::managed::async::prefetch(region, device, stream);
}

template <class Range>
void prefetch_to_host(Range&& r) {
	using T_ = typename std::remove_reference<Range>::type::value_type;
	// Due to missing const_range in cuda-api-wrappers
	using T = typename std::remove_const<T_>::type;
	cuda::memory::managed::region_t region{const_cast<T*>(r.data()), r.size() * sizeof(T)};
	cuda::memory::managed::async::prefetch_to_host(region);
}

} // namespace async

} // namespace thrustshift
