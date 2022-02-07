#pragma once

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include <cuda/runtime_api.hpp>

#include <gsl-lite/gsl-lite.hpp>

#include <cub/cub.cuh>

#include <thrustshift/math.h>
#include <thrustshift/not-a-vector.h>

namespace cub {

template <typename _UnsignedBits, typename T>
struct BaseTraits<SIGNED_INTEGER,
                  false,
                  false,
                  _UnsignedBits,
                  thrustshift::AbsView<T>> {
	typedef _UnsignedBits UnsignedBits;

	static const Category CATEGORY = SIGNED_INTEGER;
	static const UnsignedBits HIGH_BIT = UnsignedBits(1)
	                                     << ((sizeof(UnsignedBits) * 8) - 1);
	static const UnsignedBits LOWEST_KEY = HIGH_BIT;
	static const UnsignedBits MAX_KEY = UnsignedBits(-1) ^ HIGH_BIT;

	enum {
		PRIMITIVE = true,
		NULL_TYPE = false,
	};

	static __device__ __forceinline__ UnsignedBits TwiddleIn(UnsignedBits key) {
		T val = key;
		if (val >= 0) {
			return val;
		}
		UnsignedBits new_val = std::abs(val);
		return new_val | HIGH_BIT;
	};

	static __device__ __forceinline__ UnsignedBits
	TwiddleOut(UnsignedBits key) {
		if (key & HIGH_BIT) {
			auto del_high_bit = key ^ HIGH_BIT;
			T x = -T(del_high_bit);
			return x;
		}
		return key;
	};

	static __host__ __device__ __forceinline__ T Max() {
		return std::numeric_limits<T>::max();
	}

	static __host__ __device__ __forceinline__ T Lowest() {
		return std::numeric_limits<T>::lowest();
	}
};

template <typename _UnsignedBits, typename T>
struct BaseTraits<FLOATING_POINT,
                  false,
                  false,
                  _UnsignedBits,
                  thrustshift::AbsView<T>> {
	typedef _UnsignedBits UnsignedBits;

	static const Category CATEGORY = FLOATING_POINT;
	static const UnsignedBits HIGH_BIT = UnsignedBits(1)
	                                     << ((sizeof(UnsignedBits) * 8) - 1);
	static const UnsignedBits LOWEST_KEY = UnsignedBits(-1);
	static const UnsignedBits MAX_KEY = UnsignedBits(-1) ^ HIGH_BIT;

	enum {
		PRIMITIVE = true,
		NULL_TYPE = false,
	};

	static __device__ __forceinline__ UnsignedBits TwiddleIn(UnsignedBits key) {
		return key;
	};

	static __device__ __forceinline__ UnsignedBits
	TwiddleOut(UnsignedBits key) {
		return key;
	};

	static __host__ __device__ __forceinline__ T Max() {
		return FpLimits<T>::Max();
	}

	static __host__ __device__ __forceinline__ T Lowest() {
		return FpLimits<T>::Lowest();
	}
};

template <>
struct NumericTraits<thrustshift::AbsView<int>>
    : BaseTraits<SIGNED_INTEGER,
                 false,
                 false,
                 unsigned int,
                 thrustshift::AbsView<int>> {};
template <>
struct NumericTraits<thrustshift::AbsView<long int>>
    : BaseTraits<SIGNED_INTEGER,
                 false,
                 false,
                 long unsigned int,
                 thrustshift::AbsView<long int>> {};
template <>
struct NumericTraits<thrustshift::AbsView<long long int>>
    : BaseTraits<SIGNED_INTEGER,
                 false,
                 false,
                 long long unsigned int,
                 thrustshift::AbsView<long long int>> {};
template <>
struct NumericTraits<thrustshift::AbsView<float>>
    : BaseTraits<FLOATING_POINT,
                 false,
                 false,
                 unsigned int,
                 thrustshift::AbsView<float>> {};
template <>
struct NumericTraits<thrustshift::AbsView<double>>
    : BaseTraits<FLOATING_POINT,
                 false,
                 false,
                 long long unsigned int,
                 thrustshift::AbsView<double>> {};

} // namespace cub

namespace thrustshift {

namespace async {

/*! \brief Batched sort of keys with values.
 *
 *  Uses CUB's radix sort.
 */
template <class KeyInRange,
          class KeyOutRange,
          class ValueInRange,
          class ValueOutRange,
          class MemoryResource>
void sort_batched_descending(cuda::stream_t& stream,
                             KeyInRange&& keys_in,
                             KeyOutRange&& keys_out,
                             ValueInRange&& values_in,
                             ValueOutRange&& values_out,
                             std::size_t batch_len,
                             MemoryResource& delayed_memory_resource) {

	const std::size_t N = keys_in.size();

	gsl_Expects(N % batch_len == 0);
	gsl_Expects(batch_len > 0);
	gsl_Expects(keys_out.size() == N);
	gsl_Expects(values_in.size() == N);
	gsl_Expects(values_out.size() == N);

	auto cit = thrust::make_counting_iterator(0);
	auto tit = thrust::make_transform_iterator(
	    cit, [batch_len] __device__(int i) { return i * batch_len; });

	const std::size_t num_batches = N / batch_len;

	size_t tmp_bytes_size = 0;
	void* tmp_ptr = nullptr;

	using KeyT = typename std::remove_reference<KeyOutRange>::type::value_type;

	auto exec = [&] {
		cuda::throw_if_error(cub::DeviceSegmentedRadixSort::SortPairsDescending(
		    tmp_ptr,
		    tmp_bytes_size,
		    keys_in.data(),
		    keys_out.data(),
		    values_in.data(),
		    values_out.data(),
		    gsl_lite::narrow<int>(N),
		    gsl_lite::narrow<int>(num_batches),
		    tit,
		    tit + 1,
		    0, // default value by CUB
		    sizeof(KeyT) * 8, // default value by CUB
		    stream.handle()));
	};
	exec();
	auto tmp =
	    make_not_a_vector<uint8_t>(tmp_bytes_size, delayed_memory_resource);
	tmp_ptr = tmp.to_span().data();
	exec();
}

/*! \brief Batched sort of keys with respect to their absolute values.
 *
 *  Example:
 *
 *  ```
 *  batch_len = 5
 *  keys_in = {-8, 7, 10, -6, 5}
 *  // sort_batched_abs
 *  keys_out = {5, -6, 7, -8, 10}
 *  ```
 *
 *  ```
 *  batch_len = 3
 *  keys_in = {-8, 7, 10,   -6, 5, 4}
 *  // sort_batched_abs
 *  keys_out = {7, -8, 10,   4, 5, -6}
 *  ```
 */
template <class KeyInRange,
          class KeyOutRange,
          class ValueInRange,
          class ValueOutRange,
          class MemoryResource>
void sort_batched_abs(cuda::stream_t& stream,
                      KeyInRange&& keys_in,
                      KeyOutRange&& keys_out,
                      ValueInRange&& values_in,
                      ValueOutRange&& values_out,
                      std::size_t batch_len,
                      MemoryResource& delayed_memory_resource) {

	const std::size_t N = keys_in.size();

	gsl_Expects(N % batch_len == 0);
	gsl_Expects(batch_len > 0);
	gsl_Expects(keys_out.size() == N);
	gsl_Expects(values_in.size() == N);
	gsl_Expects(values_out.size() == N);

	using KeyT = typename std::remove_reference<KeyOutRange>::type::value_type;
	using AbsT = AbsView<KeyT>;

	auto cit = thrust::make_counting_iterator(0);
	auto tit = thrust::make_transform_iterator(
	    cit, [batch_len] __device__(int i) { return i * batch_len; });

	const std::size_t num_batches = N / batch_len;

	size_t tmp_bytes_size = 0;
	void* tmp_ptr = nullptr;

	auto exec = [&] {
		cuda::throw_if_error(cub::DeviceSegmentedRadixSort::SortPairs(
		    tmp_ptr,
		    tmp_bytes_size,
		    reinterpret_cast<const AbsT*>(keys_in.data()),
		    reinterpret_cast<AbsT*>(keys_out.data()),
		    values_in.data(),
		    values_out.data(),
		    gsl_lite::narrow<int>(N),
		    gsl_lite::narrow<int>(num_batches),
		    tit,
		    tit + 1,
		    0, // first key bit for comparison
		    sizeof(KeyT) * 8 - 1, // highest bit is used to save sign
		    stream.handle()));
	};
	exec();
	auto tmp =
	    make_not_a_vector<uint8_t>(tmp_bytes_size, delayed_memory_resource);
	tmp_ptr = tmp.to_span().data();
	exec();
}

} // namespace async

} // namespace thrustshift
