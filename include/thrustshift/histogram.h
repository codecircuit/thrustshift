#pragma once

#include <cub/cub.cuh>

#include <thrustshift/defines.h>
#include <thrustshift/not-a-vector.h>

namespace thrustshift {

namespace async {

template <class SampleIterator,
          class HistogramIterator,
          class I0,
          class I1,
          class I2,
          class MemoryResource>
void bin_values_into_histogram(cudaStream_t& stream,
                               SampleIterator samples,
                               HistogramIterator histogram,
                               I0 num_bins,
                               I1 lower_level,
                               I1 upper_level,
                               I2 num_samples,
                               MemoryResource& delayed_memory_resource) {

	const auto num_levels = num_bins + 1;

	size_t tmp_bytes_size = 0;
	void* tmp_ptr = nullptr;

	auto exec = [&] {
		THRUSTSHIFT_CHECK_CUDA_ERROR(
		    cub::DeviceHistogram::HistogramEven(tmp_ptr,
		                                        tmp_bytes_size,
		                                        samples,
		                                        histogram,
		                                        num_levels,
		                                        lower_level,
		                                        upper_level,
		                                        num_samples,
		                                        stream));
	};
	exec();
	auto tmp =
	    make_not_a_vector<uint8_t>(tmp_bytes_size, delayed_memory_resource);
	tmp_ptr = tmp.to_span().data();
	exec();
}

} // namespace async

} // namespace thrustshift
