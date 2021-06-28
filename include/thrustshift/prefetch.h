#pragma once

#include <type_traits>

#include <cuda/runtime_api.hpp>

namespace thrustshift {

namespace async {

template <class Range>
void prefetch(cuda::stream_t& stream, cuda::device_t& device, Range&& r) {
	using T = typename std::remove_reference<Range>::type::value_type;
	cuda::memory::managed::async::prefetch(
	    {r.data(), r.size() * sizeof(T)}, device, stream);
}

template <class Range>
void prefetch_to_host(Range&& r) {
	using T = typename std::remove_reference<Range>::type::value_type;
	cuda::memory::managed::async::prefetch_to_host(
	    {r.data(), r.size() * sizeof(T)});
}

} // namespace async

} // namespace thrustshift
