#pragma once

#include <cooperative_groups.h>

#include <gsl-lite/gsl-lite.hpp>

#include <cuda/runtime_api.hpp>

#include <cub/cub.cuh>

#include <thrustshift/constant.h>
#include <thrustshift/copy.h>
#include <thrustshift/fill.h>
#include <thrustshift/math.h>
#include <thrustshift/not-a-vector.h>
#include <thrustshift/select-if.h>
#include <thrustshift/type-traits.h>

namespace thrustshift {

namespace device_function {

namespace implicit_unroll {

/*! \brief Sum an n x N array column-wise.
 *  \param N length of row
 *  \param n number of rows
 *  \param p input of length `N * n`
 *  \param result of length `N`. As a result you may pass `p`.
 */
template <typename T0, typename T1, typename I0, typename I1, typename I2>
CUDA_FD void sum_subsequent_into(const T0* p,
                                 T1* result,
                                 int tid,
                                 I0 num_threads,
                                 I1 N,
                                 I2 n) {

	auto num_columns_per_thread = N / num_threads;
	auto sum_column = [&](int col_id) {
		T0 x{};
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
	auto num_rest = N % num_threads;
	if (tid < num_rest) {
		const int col_id = num_columns_per_thread * num_threads + tid;
		sum_column(col_id);
	}
}

} // namespace implicit_unroll

namespace explicit_unroll {

/*! \brief Sum an n x N array column-wise.
 *  \param N length of row
 *  \param n number of rows
 *  \param p input of length `N * n`
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

} // namespace explicit_unroll

/*! \brief Bin value `x` based on a subset of 8 bits into a 256 bin histogram with a full warp.
 *
 *  Only one atomic addition is executed for each bin, although multiple threads in a warp can
 *  have a value which belongs into the same bin. Via warp level primitives the bin incrementation is
 *  aggregated to one single thread.
 *
 *  ```
 *  x x x x x x x x x x x x x x x x x x x x x x x  <--- `x` represents one bit
 * |--bit_offset---|--bit_size---|--rest---------|
 * |-----prefix----|
 * ```
 *
 *  \param x value which is put into the bin. Can be different for each thread.
 *  \param bit_offset number of bits which are omitted from the left of the bit pattern of `x`.
 *  \param prefix bit pattern with `bit_offset` reasonable bits from the right. If the beginning of the bit pattern
 *      of `x` is not equal to `prefix`, the value is not binned.
 *  \param histogram length of 256. Does not have to be unique for each warp.
 *  \param valid_write set to false if a thread should not bin the value `x`.
 *  \param bin_index_transform lambda with `[] (int i) -> int {...}` signature. If set to the identity
 *      the values are binned from low to high. The lambda can be used to reverse the order in the
 *      histogram.
 */
template <typename T, typename IH, class F>
CUDA_FD void bin_value256(T x,
                          int bit_offset,
                          uint64_t prefix,
                          IH* histogram,
                          bool valid_write,
                          F bin_index_transform) {

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

	int bi = bin_index_transform(b);

	int k = 1; // increment of the bin
	// Only count values which start with the given prefix pattern
	if ((i >> sizeof(I) * 8 - bit_offset) != static_cast<I>(prefix) ||
	    !valid_write) {
		bi = -1; // value has different prefix
		k = 0;
	}

	// Threads with higher ID take precedence over those with low ID and the
	// same bin.
	const int lane_id = threadIdx.x % warp_size;

	const int mask = __match_any_sync(0xffffffff, bi) &
	                 (~(1 << lane_id)); // set our own bit to zero

	const int lsbi = sizeof(int) * 8 - __clz(mask) -
	                 1; // get ID of highest thread which has this value
	const unsigned umask =
	    *reinterpret_cast<const unsigned*>((const void*) &mask);
	k = lsbi > lane_id ? 0 : (__popc(umask) + 1);

	if (k > 0 && bi >= 0) {
		atomicAdd(histogram + bi, k);
	}
}

namespace implicit_unroll {

/*! \brief Bin transformed values based on a subset of 8 bits into a 256 bin histogram.
 *
 *  After execution and synchronization, multiple histograms (`histograms`) are filled, which must subsequently be summed into
 *  one single histogram.
 *
 *  ```
 *  x x x x x x x x x x x x x x x x x x x x x x x  <--- `x` represents one bit
 * |--bit_offset---|--bit_size---|--rest---------|
 * |-----prefix----|
 * ```
 *
 *  \param values of length N.
 *  \param histograms of length `256 * block_dim / warp_size`. Contains the result after execution and must be
 *      initialized before function execution.
 *  \param num_histograms number of histograms.
 *  \param my_tile_start is equal to zero for intra block binning and equal to `blockIdx.x` for inter block binning.
 *  \param tile_increment is equal to one for intra block binning and equal to `gridDim.x` for inter block binning.
 *      For best performance, this should be a compile time constant.
 *  \param bit_offset number of bits which are omitted from the left of the bit pattern of `x`.
 *  \param num_threads the number of threads which are entering the function (num_threads % warp_size == 0).
 *  \param prefix bit pattern with `bit_offset` reasonable bits from the right. If the beginning of the bit pattern
 *      of `x` is not equal to `prefix`, the value is not binned.
 *  \param unary_functor lambda to transform the values before binning.
 *  \param bin_index_transform lambda with `[] (int i) -> int {...}` signature. If set to the identity
 *      the values are binned from low to high. The lambda can be used to reverse the order in the
 *      histogram.
 */
template <typename T,
          typename I0,
          typename I1,
          typename I2,
          typename I3,
          typename I4,
          typename I5,
          class F0,
          class F1>
CUDA_FHD void bin_values256(
    const T* values,
    I0 N,
    I1* histograms,
    I5 num_histograms,
    I2 my_tile_start, // =0 in case of intra block, =blockIdx.x in case of inter block binning
    I3 tile_increment, // =1 in case of intra block, =grid_dim in case of inter block binning
    int tid,
    I4 num_threads,
    int bit_offset,
    uint64_t prefix,
    F0 unary_functor,
    F1 bin_index_transform) {

	gsl_ExpectsAudit(num_threads % warp_size == 0);
	constexpr int histogram_length = 256;
	const int warp_id = tid / warp_size;
	auto num_warps = num_threads / warp_size;
	auto num_warps_per_histogram = num_histograms / num_warps;
	const int histogram_id = warp_id / num_warps_per_histogram;
	I1* my_histogram = histograms + histogram_id * histogram_length;

	const int num_tiles = thrustshift::ceil_divide(N, num_threads);
	auto tile_size = num_threads;
	int tile_id = my_tile_start;
	for (; tile_id < num_tiles - 1; tile_id += tile_increment) {
		const int tile_offset = tile_id * num_threads;
		const auto x = unary_functor(values[tile_offset + tid]);
		bin_value256(
		    x, bit_offset, prefix, my_histogram, true, bin_index_transform);
	}
	// last tile
	if (tile_id == num_tiles - 1) {
		const int tile_offset = tile_id * num_threads;
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
		// `*__sync` warp level primitives.
		bin_value256(
		    x, bit_offset, prefix, my_histogram, valid_rw, bin_index_transform);
	}
}

} // namespace implicit_unroll

template <int block_dim, int num_histograms, typename IH>
struct k_largest_values_abs_block {

	static constexpr int histogram_length = 256;
	static constexpr int num_warps = block_dim / warp_size;
	static_assert(block_dim % warp_size == 0);
	static_assert(histogram_length % block_dim == 0);
	static constexpr int num_scan_elements_per_thread =
	    histogram_length / block_dim;
	using BlockLoad = cub::BlockLoad<IH,
	                                 block_dim,
	                                 num_scan_elements_per_thread,
	                                 cub::BLOCK_LOAD_WARP_TRANSPOSE>;
	using BlockScan = cub::BlockScan<IH, block_dim>;
	using BlockStore = cub::BlockStore<IH,
	                                   block_dim,
	                                   num_scan_elements_per_thread,
	                                   cub::BLOCK_STORE_WARP_TRANSPOSE>;

	struct pair_t {
		int k;
		uint64_t prefix;
	};

	union TempStorage {
		typename BlockLoad::TempStorage block_load;
		typename BlockScan::TempStorage block_scan;
		typename BlockStore::TempStorage block_store;
		pair_t pair;
	};

	template <typename T, typename I0>
	static CUDA_FD thrust::tuple<uint64_t, int>
	k_largest_values_abs_radix_block(const T* values,
	                                 I0 N,
	                                 IH* uninitialized_histograms,
	                                 int k,
	                                 TempStorage& temp_storage) {

		const int tid = threadIdx.x;

		auto unary_functor = [](T x) {
			using std::abs;
			return abs(x);
		};

		// Create the histogram from large to small values
		auto bin_index_transform = [](auto i) {
			return histogram_length - i - 1;
		};

		uint64_t prefix = 0;

		for (int bit_offset = 0; bit_offset < int(sizeof(T) * 8);
		     bit_offset += 8) {

			//
			// Initialize the histograms
			//
			device_function::implicit_unroll::fill(
			    uninitialized_histograms,
			    0,
			    tid,
			    block_dim,
			    num_warps * histogram_length);
			auto* histograms =
			    uninitialized_histograms; // rename for better readability
			__syncthreads();

			//
			// Bin values into one histogram per warp
			//
			thrustshift::device_function::implicit_unroll::bin_values256(
			    values,
			    N,
			    histograms,
			    num_histograms,
			    0, // tile start
			    1, // tile increment
			    tid,
			    block_dim,
			    bit_offset,
			    prefix,
			    unary_functor,
			    bin_index_transform);
			__syncthreads();

			//
			// Sum all histograms
			//
			if (num_histograms > 1) {
				thrustshift::device_function::implicit_unroll::
				    sum_subsequent_into(histograms,
				                        histograms,
				                        tid,
				                        block_dim,
				                        histogram_length,
				                        num_warps);
				__syncthreads();
			}
			// The first histogram is now the sum of all histograms in shared memory

			//
			// Scan the histogram
			//
			IH hvalues[num_scan_elements_per_thread];
			IH hcumulative_values[num_scan_elements_per_thread];
			BlockLoad(temp_storage.block_load).Load(histograms, hvalues);
			__syncthreads(); // necessary due to reuse of temporary memory
			BlockScan(temp_storage.block_scan)
			    .InclusiveSum(hvalues, hcumulative_values);
			__syncthreads();

			// Create helper array to have the value of our left neighbour
			IH hcumulative_values2[num_scan_elements_per_thread + 1];
#pragma unroll
			for (int j = 0; j < num_scan_elements_per_thread; ++j) {
				hcumulative_values2[j + 1] = hcumulative_values[j];
			}

			BlockStore(
			    temp_storage
			        .block_store) // block store changes the register values
			    .Store(histograms, hcumulative_values);
			__syncthreads();

			if (tid > 0) {
				hcumulative_values2[0] =
				    histograms[tid * num_scan_elements_per_thread - 1];
			}
			else {
				hcumulative_values2[0] = 0;
			}
#pragma unroll
			for (int j = 0; j < num_scan_elements_per_thread; ++j) {
				const int l = j + tid * num_scan_elements_per_thread;
				const int i = histogram_length - l - 1;
				if (hcumulative_values2[j + 1] >= k &&
				    hcumulative_values2[j] < k) {
					// Only one thread is expected to enter this branch
					prefix = (prefix << 8) | uint64_t(i);
					// all values with this prefix and larger are included
					temp_storage.pair.k = hcumulative_values2[j + 1] == k
					                          ? 0
					                          : k - hcumulative_values2[j];
					temp_storage.pair.prefix = prefix;
				}
			}

			__syncthreads();
			k = temp_storage.pair.k;
			prefix = temp_storage.pair.prefix;
			if (k == 0) {
				return {prefix, bit_offset + 8};
			}
		}
		return {prefix, sizeof(T) * 8};
	}
};

} // namespace device_function

namespace kernel {

template <typename T, int num_threads, int N, int n>
__global__ void sum_subsequent_into(const T* p, T* result) {
	const int tid = threadIdx.x;
	thrustshift::device_function::explicit_unroll::
	    sum_subsequent_into<T, num_threads, N, n>(p, result, tid);
}

template <typename T,
          int block_dim,
          int grid_dim,
          int num_sh_histograms,
          class F>
__global__ void bin_values256(const T* data,
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

	auto bin_index_transform = [](auto i) { return i; };

	device_function::explicit_unroll::
	    fill<histogram_value_type, block_dim, all_sh_histograms_length>(
	        sh_histograms, 0, tid);

	__syncthreads();

	device_function::implicit_unroll::bin_values256(data,
	                                                N,
	                                                sh_histograms,
	                                                num_sh_histograms,
	                                                blockIdx.x,
	                                                grid_dim,
	                                                tid,
	                                                block_dim,
	                                                bit_offset,
	                                                prefix,
	                                                unary_functor,
	                                                bin_index_transform);

	__syncthreads();

	//
	// Sum all histograms
	//
	if (num_sh_histograms > 1) {
		thrustshift::device_function::explicit_unroll::sum_subsequent_into<
		    histogram_value_type,
		    block_dim,
		    histogram_length,
		    num_sh_histograms>(sh_histograms, sh_histograms, tid);
		__syncthreads();
	}

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
	    kernel::
	        bin_values256<T, block_dim, num_histograms, num_sh_histograms, F>,
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
