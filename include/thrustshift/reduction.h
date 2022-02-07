#pragma once

#include <cuda/runtime_api.hpp>

#include <cub/cub.cuh>

#include <thrustshift/not-a-vector.h>

namespace thrustshift {

namespace async {

template <class Range,
          class ResultPtr,
          class ReductionF,
          class Init,
          class MemoryResource>
void reduce(cuda::stream_t& stream,
            Range&& values,
            ResultPtr result,
            ReductionF reduction_functor,
            Init initial_value,
            MemoryResource& delayed_memory_resource) {

	size_t tmp_bytes_size = 0;
	void* tmp_ptr = nullptr;
	auto exec = [&] {
		cuda::throw_if_error(
		    cub::DeviceReduce::Reduce(tmp_ptr,
		                              tmp_bytes_size,
		                              values.data(),
		                              result,
		                              gsl_lite::narrow<int>(values.size()),
		                              reduction_functor,
		                              initial_value,
		                              stream.id()));
	};
	exec();
	auto tmp =
	    make_not_a_vector<uint8_t>(tmp_bytes_size, delayed_memory_resource);
	tmp_ptr = tmp.to_span().data();
	exec();
}

template <class RangeA,
          class RangeB,
          class RangeC,
          class ReductionF,
          class Init,
          class MemoryResource>
void segmented_reduce(cuda::stream_t& stream,
                      RangeA&& values,
                      RangeB&& result,
                      RangeC&& segment_ptrs,
                      ReductionF reduction_functor,
                      Init initial_value,
                      MemoryResource& delayed_memory_resource) {

	gsl_Expects(segment_ptrs.size() > 0);

	size_t tmp_bytes_size = 0;
	void* tmp_ptr = nullptr;
	auto exec = [&] {
		cuda::throw_if_error(cub::DeviceSegmentedReduce::Reduce(
		    tmp_ptr,
		    tmp_bytes_size,
		    values.data(),
		    result.data(),
		    gsl_lite::narrow<int>(segment_ptrs.size() - 1),
		    segment_ptrs.data(),
		    segment_ptrs.data() + 1,
		    reduction_functor,
		    initial_value,
		    stream.id()));
	};
	exec();
	auto tmp =
	    make_not_a_vector<uint8_t>(tmp_bytes_size, delayed_memory_resource);
	tmp_ptr = tmp.to_span().data();
	exec();
}

template <class ItA,
          class ItB,
          class ItC,
          class ReductionF,
          class Init,
          class MemoryResource>
void segmented_reduce(cuda::stream_t& stream,
                      ItA values,
                      ItB result,
                      ItC segment_ptrs,
                      int num_segments,
                      ReductionF reduction_functor,
                      Init initial_value,
                      MemoryResource& delayed_memory_resource) {

	size_t tmp_bytes_size = 0;
	void* tmp_ptr = nullptr;
	auto exec = [&] {
		cuda::throw_if_error(
		    cub::DeviceSegmentedReduce::Reduce(tmp_ptr,
		                                       tmp_bytes_size,
		                                       values,
		                                       result,
		                                       num_segments,
		                                       segment_ptrs,
		                                       segment_ptrs + 1,
		                                       reduction_functor,
		                                       initial_value,
		                                       stream.id()));
	};
	exec();
	auto tmp =
	    make_not_a_vector<uint8_t>(tmp_bytes_size, delayed_memory_resource);
	tmp_ptr = tmp.to_span().data();
	exec();
}

} // namespace async

} // namespace thrustshift
