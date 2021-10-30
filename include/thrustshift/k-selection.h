#pragma once

#include <cooperative_groups.h>

#include <gsl-lite/gsl-lite.hpp>

#include <cuda/runtime_api.hpp>

#include <thrustshift/constant.h>
#include <thrustshift/copy.h>
#include <thrustshift/fill.h>
#include <thrustshift/math.h>
#include <thrustshift/not-a-vector.h>
#include <thrustshift/select-if.h>
#include <thrustshift/type-traits.h>

namespace thrustshift {

/*! \brief sum column-wise and save the result into the first row.
 *  \param N length of row
 *  \param n number of rows
 *  \param p input
 *  \param result of length `N`. As a result you may pass `p`.
 */
template <typename T, int num_threads, int N, int n>
CUDA_FD void sum_subsequent_into(const T* p, T* result, int tid) {

	constexpr int num_columns_per_thread = N / num_threads;
	auto sum_column = [&](int col_id) {
		T x{};
#pragma unroll
		for (int j = 0; j < n; ++j) {
			x += p[col_id + N * j];
		}
		result[col_id] = x;
	};
#pragma unroll
	for (int i = 0; i < num_columns_per_thread; ++i) {
		const int col_id = i * num_threads + tid;
		sum_column(col_id);
	}
	constexpr int num_rest = N % num_threads;
	if (tid < num_rest) {
		const int col_id = num_columns_per_thread * num_threads + tid;
		sum_column(col_id);
	}
}

template <typename T, typename IH>
CUDA_FD void bin_value(T x,
                       int bit_offset,
                       uint64_t prefix,
                       IH* histogram,
                       bool valid_write) {

	using I = typename thrustshift::make_uintegral_of_equal_size<T>::type;
	using K = uint8_t;
	constexpr int bit_size = sizeof(K) * 8;
	// extract the bits of length `bit_size` from the left with an offset of `bit_offset` to the right.
	//
	//  x x x x x x x x x x x x x x x x x x x x x x x  <--- `x` represents one bit
	// |--bit_offset---|--bit_size---|--rest---------|
	//
	const I i = *reinterpret_cast<I*>((void*) (&x));
	const K b = (i >> (sizeof(I) * 8 - bit_size - bit_offset)) &
	            I(std::numeric_limits<K>::max());

	// NOTE: in a trivial implementation the value is just incremented atomically
	// if ((i >> sizeof(I) * 8 - bit_offset) == static_cast<I>(prefix) &&
	//     valid_write) {
	// 	//atomicAdd(histogram + b, 1); // 213 GB/s
	// 	//atomicInc(histogram + b, std::numeric_limits<IH>::max()); // 212 GB/s
	// }

	int bi = b;
	int k = 1; // increment of the bin
	// Only count values which start with the given prefix pattern
	if ((i >> sizeof(I) * 8 - bit_offset) != static_cast<I>(prefix) ||
	    !valid_write) {
		bi = -1; // value has different prefix
		k = 0;
	}

	// Threads with higher ID take precedence over those with low ID and the
	// same bin.
	const int lane_id = threadIdx.x % warpSize;

	// NOTE: In the following `__match_any_sync` the threads of the current warp must
	// sync. Therefore, the read and write access in the end of the function should not
	// be suspect to a read/write hazard, as one histogram belongs to one warp. Nevertheless,
	// the compute-sanitizer reports a hazard, which is caused by a different threads processing
	// different tiles within the same warp. That can be checked by adding a synchronization within
	// the warp after this function call (`bin_value`).

	__syncwarp(); // with 264 GB/s, without 364 GB/s
	const int mask = __match_any_sync(0xffffffff, bi) & (~(1 << lane_id));
	// NOTE: There is a performance difference between __match_any_sync() and active.match_any()
	// auto active = cooperative_groups::coalesced_threads();
	//    const int mask = active.match_any(bi) & (~(1 << lane_id)); // set our own bit to zero

	const int lsbi = sizeof(int) * 8 - __clz(mask) -
	                 1; // get ID of highest thread which has this value
	const unsigned umask =
	    *reinterpret_cast<const unsigned*>((const void*) &mask);
	k = lsbi > lane_id ? 0 : (__popc(umask) + 1);

	if (k > 0 && bi >= 0) {
		histogram[bi] += k;
	}
}

//! one warp for each histogram
template <typename T,
          typename I0,
          typename I1,
          typename I2,
          typename I3,
          typename I4,
          class F>
CUDA_FHD void bin_values_block(
    const T* values,
    I0 N,
    I1* histograms,
    I2 my_tile_start, // =0 in case of intra block, =blockIdx.x in case of inter block binning
    I3 tile_increment, // =1 in case of intra block, =grid_dim in case of inter block binning
    I4 block_dim,
    int bit_offset,
    uint64_t prefix,
    F unary_functor) {

	constexpr int histogram_length = 256;
	const int tid = threadIdx.x;
	const int warp_id = tid / warp_size;
	I1* my_histogram = histograms + warp_id * histogram_length;

	const int num_tiles = thrustshift::ceil_divide(N, block_dim);
	auto tile_size = block_dim;
	int tile_id = my_tile_start;
	for (; tile_id < num_tiles - 1; tile_id += tile_increment) {
		const int tile_offset = tile_id * block_dim;
		const auto x = unary_functor(values[tile_offset + tid]);
		bin_value(x, bit_offset, prefix, my_histogram, true);
	}
	// last tile
	if (tile_id == num_tiles - 1) {
		const int tile_offset = tile_id * block_dim;
		const int curr_tile_size =
		    tile_offset + tile_size > N ? (N - tile_offset) : tile_size;
		bool valid_rw = tid < curr_tile_size;
		const auto x = [&] {
			if (valid_rw) {
				return unary_functor(values[tile_offset + tid]);
			}
			return T{};
		}();
		// every thread of the warp must enter this function as it contains
		// warp synchronizing functions.
		bin_value(x, bit_offset, prefix, my_histogram, valid_rw);
	}
}

namespace kernel {

template <typename T, int num_threads, int N, int n>
__global__ void sum_subsequent_into(const T* p, T* result) {
	const int tid = threadIdx.x;
	thrustshift::sum_subsequent_into<T, num_threads, N, n>(p, result, tid);
}

template <typename T,
          int block_dim,
          int grid_dim,
          int num_sh_histograms,
          class F>
__global__ void bin_values(const T* data,
                           int N,
                           int* histograms,
                           int bit_offset,
                           uint64_t prefix,
                           F unary_functor) {

	constexpr int histogram_length = 256;
	constexpr int all_sh_histograms_length =
	    histogram_length * num_sh_histograms;

	using histogram_value_type = unsigned;
	__shared__ histogram_value_type sh_histograms[all_sh_histograms_length];
	const int tid = threadIdx.x;
	constexpr int num_warps = block_dim / warp_size;
	static_assert(num_sh_histograms == num_warps);

	auto cta = cooperative_groups::this_thread_block();

	fill_unroll<histogram_value_type, block_dim, all_sh_histograms_length>(
	    sh_histograms, 0, tid);

	cta.sync();

	bin_values_block(data,
	                 N,
	                 sh_histograms,
	                 blockIdx.x,
	                 grid_dim,
	                 block_dim,
	                 bit_offset,
	                 prefix,
	                 unary_functor);

	cta.sync();

	//
	// Sum all histograms
	//
	thrustshift::sum_subsequent_into<histogram_value_type,
	                                 block_dim,
	                                 histogram_length,
	                                 num_sh_histograms>(
	    sh_histograms, sh_histograms, tid);
	cta.sync();

	thrustshift::block_copy<block_dim, histogram_length>(
	    sh_histograms, histograms + blockIdx.x * histogram_length);
}

} // namespace kernel

namespace async {

template <typename T, class MemoryResource, class F>
void bin_values256(cuda::stream_t& stream,
                   gsl_lite::span<const T> values,
                   gsl_lite::span<int> histogram,
                   int bit_offset,
                   uint64_t prefix,
                   F unary_functor,
                   MemoryResource& delayed_memory_resource) {

	constexpr int histogram_length = 256;
	gsl_Expects(histogram.size() == histogram_length);

	constexpr int block_dim = 256;
	constexpr int num_histograms = 80;
	constexpr int num_sh_histograms = block_dim / warp_size;

	auto c = cuda::make_launch_config(num_histograms, block_dim);

	auto tmp_mem = make_not_a_vector<int>(num_histograms * histogram_length,
	                                      delayed_memory_resource);

	auto histograms = tmp_mem.to_span();

	cuda::enqueue_launch(
	    kernel::bin_values<T, block_dim, num_histograms, num_sh_histograms, F>,
	    stream,
	    c,
	    values.data(),
	    values.size(),
	    histograms.data(),
	    bit_offset,
	    prefix,
	    unary_functor);

	cuda::enqueue_launch(kernel::sum_subsequent_into<int,
	                                                 block_dim,
	                                                 histogram_length,
	                                                 num_histograms>,
	                     stream,
	                     cuda::make_launch_config(1, 256),
	                     histograms.data(),
	                     histogram.data());
}

} // namespace async

template <typename T, class MemoryResource>
std::tuple<uint64_t, int> k_largest_values_abs_radix(
    cuda::stream_t& stream,
    gsl_lite::span<const T> values,
    int k,
    MemoryResource& delayed_memory_resource) {

	auto unary_functor = [] __device__(T x) {
		using std::abs;
		return abs(x);
	};

	constexpr int histogram_length = 256;
	auto tmp =
	    make_not_a_vector<int>(histogram_length, delayed_memory_resource);
	auto histogram = tmp.to_span();

	uint64_t prefix = 0;
	for (int bit_offset = 0; bit_offset < int(sizeof(T) * 8); bit_offset += 8) {
		async::bin_values256<T>(stream,
		                        values,
		                        histogram,
		                        bit_offset,
		                        prefix,
		                        unary_functor,
		                        delayed_memory_resource);
		stream.synchronize();
		int acc = histogram[histogram_length - 1];
		int acc_prev = 0;
		for (int i = histogram_length - 2; i >= 0; --i) {
			acc += histogram[i];
			if (acc >= k) {
				prefix = (prefix << 8) | uint64_t(i);
				if (acc == k) {
					// all values with this prefix and larger are included
					return {prefix, bit_offset + 8};
				}
				k = k - acc_prev;
				break;
			}
			acc_prev = acc;
		}
	}
	return {prefix, sizeof(T) * 8};
}

// selected_values.size() == values.size() because CUB might select more values, if e.g. values are all equal
template <typename T, class MemoryResource>
void select_k_largest_values_abs(
    cuda::stream_t& stream,
    gsl_lite::span<const T> values,
    gsl_lite::span<thrust::tuple<T, int>> selected_values,
    int k,
    MemoryResource& delayed_memory_resource) {

	const auto tup = thrustshift::k_largest_values_abs_radix<T>(
	    stream, values, k, delayed_memory_resource);
	const auto prefix = std::get<0>(tup);
	const auto bit_offset = std::get<1>(tup);
	auto select_op = [prefix,
	                  bit_offset] __device__(const thrust::tuple<T, int>& tup) {
		using std::abs;
		auto x = abs(thrust::get<0>(tup));
		using I = typename thrustshift::make_uintegral_of_equal_size<T>::type;
		const I i = *reinterpret_cast<I*>((void*) (&x));
		return (i >> sizeof(I) * 8 - bit_offset) >= static_cast<I>(prefix);
	};
	auto tmp = make_not_a_vector<int>(1, delayed_memory_resource);
	auto num_selected = tmp.to_span();

	async::select_if_with_index(stream,
	                            values,
	                            selected_values,
	                            num_selected.data(),
	                            select_op,
	                            delayed_memory_resource);
	stream.synchronize();
}

} // namespace thrustshift
