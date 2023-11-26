#pragma once

#include <cooperative_groups.h>

#include <gsl-lite/gsl-lite.hpp>

#include <cuda/runtime_api.hpp>

#include <cub/cub.cuh>

#include <makeshift/variant.hpp>

#include <thrustshift/constant.h>
#include <thrustshift/copy.h>
#include <thrustshift/fill.h>
#include <thrustshift/histogram.h>
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

#if __CUDA_ARCH__ >= 700
	const int mask = __match_any_sync(0xffffffff, bi) &
	                 (~(1 << lane_id)); // set our own bit to zero
#else
	//
	// Naiive match_any_sync implementation for sm_xx < sm_70
	//
	int mask{};
	constexpr int warpSize = 32;
	for (int other_lane_id = 0; other_lane_id < warpSize; ++other_lane_id) {
		const int other_bi = __shfl_sync(0xffffffff, bi, other_lane_id);
		mask |= ((other_bi == bi ? 1 : 0) << other_lane_id);
	}
	mask &= (~(1 << lane_id)); // set our own bit to zero
#endif

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
	gsl_ExpectsAudit(num_warps % num_histograms == 0);
	gsl_ExpectsAudit(num_histograms <= num_warps);
	auto num_warps_per_histogram = num_warps / num_histograms;
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

	struct triplet_t {
		int k;
		uint64_t prefix;
		int selected_values_pos;
	};

	struct TempStorage {
		union {
			typename BlockLoad::TempStorage block_load;
			typename BlockScan::TempStorage block_scan;
			typename BlockStore::TempStorage block_store;
		};
		triplet_t triplet;
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
					temp_storage.triplet.k = hcumulative_values2[j + 1] == k
					                             ? 0
					                             : k - hcumulative_values2[j];
					temp_storage.triplet.prefix = prefix;
					// This is set for subsequent function execution to avoid one
					// additional __syncthreads();
					temp_storage.triplet.selected_values_pos = 0;
				}
			}

			__syncthreads();
			k = temp_storage.triplet.k;
			prefix = temp_storage.triplet.prefix;
			if (k == 0) {
				return {prefix, bit_offset + 8};
			}
		}
		return {prefix, sizeof(T) * 8};
	}

	template <typename It, typename ItSelected, typename I0>
	static CUDA_FD void select_k_largest_values_with_index_abs(
	    It values,
	    I0 N,
	    ItSelected selected_values,
	    IH* uninitialized_histograms,
	    int k,
	    TempStorage& temp_storage) {

		const int tid = threadIdx.x;

		auto tup = k_largest_values_abs_radix_block(
		    values, N, uninitialized_histograms, k, temp_storage);
		auto prefix = thrust::get<0>(tup);
		auto bit_offset = thrust::get<1>(tup);

		using T = typename std::remove_const<
		    typename std::iterator_traits<It>::value_type>::type;

		auto cit = thrust::make_counting_iterator(0);
		auto it = thrust::make_zip_iterator(thrust::make_tuple(values, cit));

		auto select_op = [prefix, bit_offset] __host__ __device__(
		                     const thrust::tuple<T, int>& tup) {
			using std::abs;
			auto x = abs(thrust::get<0>(tup));
			using I =
			    typename thrustshift::make_uintegral_of_equal_size<T>::type;
			const I i = *reinterpret_cast<I*>((void*) (&x));
			return (i >> sizeof(I) * 8 - bit_offset) >= static_cast<I>(prefix);
		};

		device_function::implicit_unroll::select_if(
		    it,
		    N,
		    selected_values,
		    &temp_storage.triplet.selected_values_pos,
		    tid,
		    block_dim,
		    select_op);
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
          class F0,
          class F1>
__global__ void bin_values256(const T* data,
                              int N,
                              int* histograms,
                              int bit_offset,
                              uint64_t prefix,
                              F0 unary_functor,
                              F1 bin_index_transform) {

	constexpr int histogram_length = 256;
	constexpr int all_sh_histograms_length =
	    histogram_length * num_sh_histograms;

	using histogram_value_type = unsigned;
	__shared__ histogram_value_type sh_histograms[all_sh_histograms_length];
	const int tid = threadIdx.x;

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

template <typename T,
          int block_dim,
          int grid_dim,
          int num_sh_histograms,
          class F0,
          class F1>
__global__ void bin_values256_threadfence(const T* data,
                                          int N,
                                          volatile int* histograms,
                                          int bit_offset,
                                          uint64_t prefix,
                                          F0 unary_functor,
                                          F1 bin_index_transform,
                                          unsigned* entry_ticket,
                                          unsigned* exit_ticket) {

	constexpr int histogram_length = 256;
	constexpr int all_sh_histograms_length =
	    histogram_length * num_sh_histograms;

	using histogram_value_type = int;
	__shared__ histogram_value_type sh_histograms[all_sh_histograms_length];
	const int tid = threadIdx.x;

	device_function::explicit_unroll::
	    fill<histogram_value_type, block_dim, all_sh_histograms_length>(
	        sh_histograms, 0, tid);
	__shared__ int sh_entry_bid;
	__shared__ int sh_exit_bid;
	if (tid == 0) {
		sh_entry_bid = atomicInc(entry_ticket, grid_dim);
	}
	__syncthreads();
	int bid = sh_entry_bid;

	device_function::implicit_unroll::bin_values256(data,
	                                                N,
	                                                sh_histograms,
	                                                num_sh_histograms,
	                                                bid,
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
	    sh_histograms, histograms + bid * histogram_length);

	// To my understanding this sync threads is necessary because it might happen that
	// thread 0 does not observe the write of all threads of our histogram.
	__syncthreads();

	__threadfence();

	if (tid == 0) {
		sh_exit_bid = atomicInc(exit_ticket, grid_dim);
	}
	__syncthreads();
	if (sh_exit_bid == grid_dim - 1) {
		static_assert(block_dim >= histogram_length);
		if (tid < histogram_length) {
			histogram_value_type h = 0;
#pragma unroll
			for (int histo_id = 0; histo_id < grid_dim; ++histo_id) {
				h += histograms[histo_id * histogram_length + tid];
			}
			histograms[tid] = h;
		}
	}
}

template <typename T,
          int block_dim,
          int grid_dim,
          int num_sh_histograms,
          class F0,
          class F1>
__global__ void bin_values256_atomic(const T* data,
                                     int N,
                                     int* histogram,
                                     int bit_offset,
                                     uint64_t prefix,
                                     F0 unary_functor,
                                     F1 bin_index_transform) {

	constexpr int histogram_length = 256;
	constexpr int all_sh_histograms_length =
	    histogram_length * num_sh_histograms;

	using histogram_value_type = int;

	__shared__ histogram_value_type sh_histograms[all_sh_histograms_length];

	const int tid = threadIdx.x;
	const int bid = blockIdx.x;

	device_function::explicit_unroll::
	    fill<histogram_value_type, block_dim, all_sh_histograms_length>(
	        sh_histograms, 0, tid);

	__syncthreads();

	device_function::implicit_unroll::bin_values256(data,
	                                                N,
	                                                sh_histograms,
	                                                num_sh_histograms,
	                                                bid,
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

	if (tid < histogram_length) {
		atomicAdd(histogram + tid, sh_histograms[tid]);
	}
}

template <typename T,
          int block_dim,
          int grid_dim,
          int num_sh_histograms,
          bool use_k0_and_zero_prefix,
          class F0,
          class F1>
__global__ void bin_values256_atomic_with_ptr(const T* data,
                                              int N,
                                              int* histogram,
                                              int bit_offset,
                                              uint64_t* prefix_,
                                              int* k_,
                                              int k0,
                                              F0 unary_functor,
                                              F1 bin_index_transform) {

	constexpr int histogram_length = 256;
	constexpr int all_sh_histograms_length =
	    histogram_length * num_sh_histograms;

	using histogram_value_type = int;

	__shared__ histogram_value_type sh_histograms[all_sh_histograms_length];

	const int tid = threadIdx.x;
	const int bid = blockIdx.x;

	const int k = [&] {
		if constexpr (use_k0_and_zero_prefix) {
			return k0;
		}
		else {
			return *k_;
		}
	}();
	if (k <= 0) {
		return;
	}

	const uint64_t prefix = [&]() -> uint64_t {
		if constexpr (use_k0_and_zero_prefix) {
			return 0;
		}
		else {
			return *prefix_;
		}
	}();

	device_function::explicit_unroll::
	    fill<histogram_value_type, block_dim, all_sh_histograms_length>(
	        sh_histograms, 0, tid);

	__syncthreads();

	device_function::implicit_unroll::bin_values256(data,
	                                                N,
	                                                sh_histograms,
	                                                num_sh_histograms,
	                                                bid,
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

	if (tid < histogram_length) {
		atomicAdd(histogram + tid, sh_histograms[tid]);
	}
}

template <typename T,
          typename IH,
          int block_dim,
          int grid_dim,
          int num_sh_histograms>
__global__ void k_select_radix(const T* data,
                               int N,
                               IH* histograms,
                               int* bit_offset_,
                               uint64_t* prefix_,
                               int* k_) {

	constexpr int histogram_length = 256;
	constexpr int all_sh_histograms_length =
	    histogram_length * num_sh_histograms;

	auto unary_functor = [](T x) {
		using std::abs;
		return abs(x);
	};

	using BlockScan = cub::BlockScan<IH, block_dim>;

	__shared__ union {
		typename BlockScan::TempStorage block_scan;
		IH histograms[all_sh_histograms_length];
	} temp_storage;

	const int tid = threadIdx.x;
	const int bid = blockIdx.x;

	uint64_t prefix;
	int k;

	auto grid = cooperative_groups::this_grid();

	auto bin_index_transform = [](auto i) { return histogram_length - i - 1; };

	for (int bit_offset = 0; bit_offset < int(sizeof(T) * 8); bit_offset += 8) {

		k = *k_;
		prefix = *prefix_;
		if (k == 0) {
			return;
		}

		device_function::explicit_unroll::
		    fill<IH, block_dim, all_sh_histograms_length>(
		        temp_storage.histograms, 0, tid);

		__syncthreads();

		device_function::implicit_unroll::bin_values256(data,
		                                                N,
		                                                temp_storage.histograms,
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
			    IH,
			    block_dim,
			    histogram_length,
			    num_sh_histograms>(
			    temp_storage.histograms, temp_storage.histograms, tid);
			__syncthreads();
		}

		IH h = temp_storage.histograms[tid];
		IH hcum;
		// it is faster if all blocks do already the scan before
		// the histograms are summed up.
		BlockScan(temp_storage.block_scan).InclusiveSum(h, hcum);
		histograms[bid * histogram_length + tid] = hcum;

		//
		// Store block histogram to global memory
		//
		// thrustshift::block_copy<block_dim, histogram_length>(
		//     temp_storage.histograms, histograms + bid * histogram_length);

		grid.sync();
		if (bid == 0) {
			static_assert(histogram_length == block_dim);
#pragma unroll
			for (int histo_id = 1; histo_id < grid_dim; ++histo_id) {
				hcum += histograms[histo_id * histogram_length + tid];
			}
			//			BlockScan(temp_storage.block_scan).InclusiveSum(h, hcum);
			//			__syncthreads();
			temp_storage.histograms[tid] = hcum;
			__syncthreads();
			IH hleft = 0;
			if (tid > 0) {
				hleft = temp_storage.histograms[tid - 1];
			}

			const int i = histogram_length - tid - 1;
			if (hcum >= k && hleft < k) {
				// Only one thread is expected to enter this branch
				prefix = (prefix << 8) | uint64_t(i);
				// all values with this prefix and larger are included
				*k_ = hcum == k ? 0 : k - hleft;
				*prefix_ = prefix;
				*bit_offset_ = bit_offset + 8;
			}

			__syncthreads();
		}
		grid.sync();
	}
}

// The summation is done redundantly
template <typename T,
          typename IH,
          int block_dim,
          int grid_dim,
          int num_sh_histograms>
__global__ void k_select_radix2(const T* data,
                                int N,
                                IH* histograms,
                                int* bit_offset_,
                                uint64_t* prefix_,
                                int k) {

	constexpr int histogram_length = 256;
	constexpr int all_sh_histograms_length =
	    histogram_length * num_sh_histograms;

	auto unary_functor = [](T x) {
		using std::abs;
		return abs(x);
	};

	using BlockScan = cub::BlockScan<IH, block_dim>;

	__shared__ union {
		typename BlockScan::TempStorage block_scan;
		IH histograms[all_sh_histograms_length];
	} temp_storage;

	const int tid = threadIdx.x;
	const int bid = blockIdx.x;

	__shared__ uint64_t sh_prefix;
	__shared__ int sh_k;

	uint64_t prefix = 0;

	auto grid = cooperative_groups::this_grid();

	auto bin_index_transform = [](auto i) { return histogram_length - i - 1; };

	for (int bit_offset = 0, l = 0; bit_offset < int(sizeof(T) * 8);
	     bit_offset += 8, ++l) {

		if (k == 0) {
			return;
		}

		device_function::explicit_unroll::
		    fill<IH, block_dim, all_sh_histograms_length>(
		        temp_storage.histograms, 0, tid);

		__syncthreads();

		device_function::implicit_unroll::bin_values256(data,
		                                                N,
		                                                temp_storage.histograms,
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
			    IH,
			    block_dim,
			    histogram_length,
			    num_sh_histograms>(
			    temp_storage.histograms, temp_storage.histograms, tid);
			__syncthreads();
		}

		IH h = temp_storage.histograms[tid];
		IH hcum;
		// it is faster if all blocks do already the scan before
		// the histograms are summed up.
		BlockScan(temp_storage.block_scan).InclusiveSum(h, hcum);
		histograms[l * grid_dim * histogram_length + bid * histogram_length +
		           tid] = hcum;

		//
		// Store block histogram to global memory
		//
		// thrustshift::block_copy<block_dim, histogram_length>(
		//     temp_storage.histograms, histograms + bid * histogram_length);

		grid.sync();
		static_assert(histogram_length == block_dim);
#pragma unroll
		for (int histo_id = 1; histo_id < grid_dim; ++histo_id) {
			hcum += histograms[l * grid_dim * histogram_length +
			                   histo_id * histogram_length + tid];
		}
		//			BlockScan(temp_storage.block_scan).InclusiveSum(h, hcum);
		//			__syncthreads();
		temp_storage.histograms[tid] = hcum;
		__syncthreads();
		IH hleft = 0;
		if (tid > 0) {
			hleft = temp_storage.histograms[tid - 1];
		}

		const int i = histogram_length - tid - 1;
		if (hcum >= k && hleft < k) {
			// Only one thread is expected to enter this branch
			prefix = (prefix << 8) | uint64_t(i);
			// all values with this prefix and larger are included
			sh_k = hcum == k ? 0 : k - hleft;
			sh_prefix = prefix;
			if (bid == 0) {
				*prefix_ = prefix;
				*bit_offset_ = bit_offset + 8;
			}
		}

		__syncthreads();
		k = sh_k;
		prefix = sh_prefix;
	}
}

template <typename T,
          typename IH,
          int block_dim,
          int num_sh_histograms,
          int num_gl_histograms>
__global__ void k_select_radix_dynamic_parallelism(const T* data,
                                                   int N,
                                                   IH* histograms,
                                                   int* bit_offset_,
                                                   uint64_t* prefix_,
                                                   int k) {

	constexpr int histogram_length = 256;

	auto unary_functor = [](T x) {
		using std::abs;
		return abs(x);
	};

	using BlockScan = cub::BlockScan<IH, block_dim>;

	__shared__ union {
		typename BlockScan::TempStorage block_scan;
		IH histogram[histogram_length];
	} temp_storage;

	__shared__ int sh_k;
	__shared__ uint64_t sh_prefix;

	static_assert(block_dim == 256);

	constexpr int child_grid_dim = num_gl_histograms;
	constexpr int child_block_dim = 256;

	const int tid = threadIdx.x;

	uint64_t prefix = 0;

	auto bin_index_transform = [](auto i) { return histogram_length - i - 1; };

#pragma unroll
	for (int bit_offset = 0; bit_offset < int(sizeof(T) * 8); bit_offset += 8) {

		if (k == 0) {
			return;
		}

		if (tid == 0) {

			kernel::bin_values256<T,
			                      child_block_dim,
			                      child_grid_dim,
			                      num_sh_histograms>
			    <<<child_grid_dim, child_block_dim>>>(data,
			                                          N,
			                                          histograms,
			                                          bit_offset,
			                                          prefix,
			                                          unary_functor,
			                                          bin_index_transform);
			cudaDeviceSynchronize();
		}
		__syncthreads();

		IH h = 0;
		IH hcum;

#pragma unroll
		for (int histo_id = 0; histo_id < num_gl_histograms; ++histo_id) {
			h += histograms[histo_id * histogram_length + tid];
		}

		BlockScan(temp_storage.block_scan).InclusiveSum(h, hcum);
		__syncthreads();

		temp_storage.histogram[tid] = hcum;
		__syncthreads();

		IH hleft = 0;
		if (tid > 0) {
			hleft = temp_storage.histogram[tid - 1];
		}

		if (hcum >= k && hleft < k) {
			const int i = histogram_length - tid - 1;
			// Only one thread is expected to enter this branch
			prefix = (prefix << 8) | uint64_t(i);
			// all values with this prefix and larger are included
			k = hcum == k ? 0 : k - hleft;

			// Write to shared memory
			sh_k = k;
			sh_prefix = prefix;

			// Write to global memory
			//*k_ = k;
			*prefix_ = prefix;
			*bit_offset_ = bit_offset + 8;
		}
		__syncthreads();
		k = sh_k;
		prefix = sh_prefix;
	}
}

template <typename T,
          typename IH,
          int block_dim,
          int num_sh_histograms,
          int child_grid_dim>
__global__ void k_select_radix_dynamic_parallelism_atomic_binning(
    const T* data,
    int N,
    IH* histogram,
    int* bit_offset_,
    uint64_t* prefix_,
    int k) {

	constexpr int histogram_length = 256;

	auto unary_functor = [](T x) {
		using std::abs;
		return abs(x);
	};

	using BlockScan = cub::BlockScan<IH, block_dim>;

	__shared__ union {
		typename BlockScan::TempStorage block_scan;
		IH histogram[histogram_length];
	} temp_storage;

	__shared__ int sh_k;
	__shared__ uint64_t sh_prefix;

	static_assert(block_dim == 256);

	constexpr int child_block_dim = 256;

	const int tid = threadIdx.x;
	const int bid = blockIdx.x;
	gsl_ExpectsAudit(bid == 0);

	uint64_t prefix = 0;

	auto bin_index_transform = [](auto i) { return histogram_length - i - 1; };

#pragma unroll
	for (int bit_offset = 0; bit_offset < int(sizeof(T) * 8); bit_offset += 8) {

		if (k == 0) {
			return;
		}

		histogram[tid] = 0;
		__syncthreads();

		if (tid == 0) {

			kernel::bin_values256_atomic<T,
			                             child_block_dim,
			                             child_grid_dim,
			                             num_sh_histograms>
			    <<<child_grid_dim, child_block_dim>>>(data,
			                                          N,
			                                          histogram,
			                                          bit_offset,
			                                          prefix,
			                                          unary_functor,
			                                          bin_index_transform);
			cudaDeviceSynchronize();
		}
		__syncthreads();

		IH h = histogram[tid];
		IH hcum;

		BlockScan(temp_storage.block_scan).InclusiveSum(h, hcum);
		__syncthreads();

		temp_storage.histogram[tid] = hcum;
		__syncthreads();

		IH hleft = 0;
		if (tid > 0) {
			hleft = temp_storage.histogram[tid - 1];
		}

		if (hcum >= k && hleft < k) {
			const int i = histogram_length - tid - 1;
			// Only one thread is expected to enter this branch
			prefix = (prefix << 8) | uint64_t(i);
			// all values with this prefix and larger are included
			k = hcum == k ? 0 : k - hleft;

			// Write to shared memory
			sh_k = k;
			sh_prefix = prefix;

			// Write to global memory
			//*k_ = k;
			*prefix_ = prefix;
			*bit_offset_ = bit_offset + 8;
		}
		__syncthreads();
		k = sh_k;
		prefix = sh_prefix;
	}
}

template <typename IH, int block_dim>
__global__ void k_select_radix_from_histogram(IH* histogram,
                                              int bit_offset,
                                              uint64_t* prefix_, // only output
                                              int* k_) {

	constexpr int histogram_length = 256;

	using BlockScan = cub::BlockScan<IH, block_dim>;

	__shared__ union {
		typename BlockScan::TempStorage block_scan;
		IH histogram[histogram_length];
	} temp_storage;

	static_assert(block_dim == 256);

	const int tid = threadIdx.x;

	uint64_t prefix = 0;
	int k = *k_;

	IH h = histogram[tid];
	IH hcum;

	BlockScan(temp_storage.block_scan).InclusiveSum(h, hcum);
	__syncthreads();

	temp_storage.histogram[tid] = hcum;
	__syncthreads();

	IH hleft = 0;
	if (tid > 0) {
		hleft = temp_storage.histogram[tid - 1];
	}

	if (hcum >= k && hleft < k) {
		const int i = histogram_length - tid - 1;
		// Only one thread is expected to enter this branch
		prefix = uint64_t(i);
		// all values with this prefix and larger are included
		k = hcum == k ? 0 : k - hleft;

		*k_ = k;
		*prefix_ = prefix;
	}
}

template <typename IH, int block_dim, bool initialize_offset_prefix_k>
__global__ void k_select_radix_from_histogram_with_ptr(IH* histogram,
                                                       int* bit_offset_,
                                                       uint64_t* prefix_,
                                                       int* k_,
                                                       int k0) {

	constexpr int histogram_length = 256;

	using BlockScan = cub::BlockScan<IH, block_dim>;

	__shared__ union {
		typename BlockScan::TempStorage block_scan;
		IH histogram[histogram_length];
	} temp_storage;

	static_assert(block_dim == 256);

	const int tid = threadIdx.x;

	uint64_t prefix = [&]() -> uint64_t {
		if constexpr (initialize_offset_prefix_k) {
			return 0;
		}
		else {
			return *prefix_;
		}
	}();
	int k = [&] {
		if constexpr (initialize_offset_prefix_k) {
			return k0;
		}
		else {
			return *k_;
		}
	}();
	int bit_offset = [&] {
		if constexpr (initialize_offset_prefix_k) {
			return 0;
		}
		else {
			return *bit_offset_;
		}
	}();

	if (k <= 0) {
		return;
	}

	IH h = histogram[tid];
	IH hcum;

	BlockScan(temp_storage.block_scan).InclusiveSum(h, hcum);
	__syncthreads();

	temp_storage.histogram[tid] = hcum;
	__syncthreads();

	IH hleft = 0;
	if (tid > 0) {
		hleft = temp_storage.histogram[tid - 1];
	}

	if (hcum >= k && hleft < k) {
		const int i = histogram_length - tid - 1;
		// Only one thread is expected to enter this branch
		prefix = (prefix << 8) | uint64_t(i);
		// all values with this prefix and larger are included
		k = hcum == k ? 0 : k - hleft;

		*k_ = k;
		*prefix_ = prefix;
		*bit_offset_ = bit_offset + 8;
	}
}

} // namespace kernel

namespace async {

template <typename T, class MemoryResource, class F0, class F1>
void bin_values256(cuda::stream_t& stream,
                   gsl_lite::span<const T> values,
                   gsl_lite::span<int> histogram,
                   int bit_offset,
                   uint64_t prefix,
                   F0 unary_functor,
                   F1 bin_index_transform,
                   MemoryResource& delayed_memory_resource) {

	constexpr int histogram_length = 256;
	gsl_Expects(histogram.size() == histogram_length);

	constexpr int block_dim = 256;
	// RTX 2080 Ti has 68 SMs, `4*` to get best occupancy in practice
	constexpr int num_histograms = 4 * 68;
	constexpr int num_sh_histograms = 1;

	auto c = cuda::make_launch_config(num_histograms, block_dim);

	auto tmp_mem = make_not_a_vector<int>(num_histograms * histogram_length,
	                                      delayed_memory_resource);

	std::cout << "bit_offset = " << bit_offset << ", prefix = " << prefix
	          << std::endl;

	auto histograms = tmp_mem.to_span();

	cuda::enqueue_launch(kernel::bin_values256<T,
	                                           block_dim,
	                                           num_histograms,
	                                           num_sh_histograms,
	                                           F0,
	                                           F1>,
	                     stream,
	                     c,
	                     values.data(),
	                     values.size(),
	                     histograms.data(),
	                     bit_offset,
	                     prefix,
	                     unary_functor,
	                     bin_index_transform);

	cuda::enqueue_launch(kernel::sum_subsequent_into<int,
	                                                 block_dim,
	                                                 histogram_length,
	                                                 num_histograms>,
	                     stream,
	                     cuda::make_launch_config(1, 256),
	                     histograms.data(),
	                     histogram.data());
}

template <typename T, class MemoryResource, class F0, class F1>
void bin_values256(cuda::stream_t& stream,
                   gsl_lite::span<const T> values,
                   gsl_lite::span<int> histogram,
                   int bit_offset,
                   uint64_t prefix,
                   int block_dim,
                   int num_histograms,
                   int num_sh_histograms,
                   F0 unary_functor,
                   F1 bin_index_transform,
                   MemoryResource& delayed_memory_resource) {

	constexpr int histogram_length = 256;
	gsl_Expects(histogram.size() == histogram_length);

	auto block_dim_v = makeshift::expand(
	    block_dim, MAKESHIFT_CONSTVAL(std::array{64, 128, 256, 512}));
	auto num_histograms_v = makeshift::expand(
	    num_histograms,
	    MAKESHIFT_CONSTVAL(std::array{68, 2 * 68, 3 * 68, 4 * 68, 8 * 68}));
	auto num_sh_histograms_v = makeshift::expand(
	    num_sh_histograms, MAKESHIFT_CONSTVAL(std::array{1, 2, 3, 4, 8}));

	std::visit(
	    [&](auto block_dim, auto num_histograms, auto num_sh_histograms) {
		    auto num_warps = block_dim / warp_size;
		    gsl_Expects(num_warps % num_sh_histograms == 0);
		    gsl_Expects(num_warps >= num_sh_histograms);
		    auto c =
		        cuda::make_launch_config(int(num_histograms), int(block_dim));

		    auto tmp_mem = make_not_a_vector<int>(
		        num_histograms * histogram_length, delayed_memory_resource);

		    auto histograms = tmp_mem.to_span();

		    cuda::enqueue_launch(kernel::bin_values256<T,
		                                               block_dim,
		                                               num_histograms,
		                                               num_sh_histograms,
		                                               F0,
		                                               F1>,
		                         stream,
		                         c,
		                         values.data(),
		                         values.size(),
		                         histograms.data(),
		                         bit_offset,
		                         prefix,
		                         unary_functor,
		                         bin_index_transform);

		    cuda::enqueue_launch(kernel::sum_subsequent_into<int,
		                                                     block_dim,
		                                                     histogram_length,
		                                                     num_histograms>,
		                         stream,
		                         cuda::make_launch_config(1, 256),
		                         histograms.data(),
		                         histogram.data());
	    },
	    block_dim_v,
	    num_histograms_v,
	    num_sh_histograms_v);
}

template <typename T, class MemoryResource, class F0, class F1>
void bin_values256_threadfence(cuda::stream_t& stream,
                               gsl_lite::span<const T> values,
                               gsl_lite::span<int> histogram,
                               int bit_offset,
                               uint64_t prefix,
                               int block_dim,
                               int num_histograms,
                               int num_sh_histograms,
                               F0 unary_functor,
                               F1 bin_index_transform,
                               MemoryResource& delayed_memory_resource) {

	constexpr int histogram_length = 256;
	gsl_Expects(histogram.size() == histogram_length);

	auto block_dim_v =
	    makeshift::expand(block_dim, MAKESHIFT_CONSTVAL(std::array{256, 512}));
	auto num_histograms_v = makeshift::expand(
	    num_histograms,
	    MAKESHIFT_CONSTVAL(std::array{68, 2 * 68, 3 * 68, 4 * 68, 8 * 68}));
	auto num_sh_histograms_v = makeshift::expand(
	    num_sh_histograms, MAKESHIFT_CONSTVAL(std::array{1, 2, 3, 4, 8}));

	auto tmp = make_not_a_vector<unsigned>(2, delayed_memory_resource);
	auto tickets = tmp.to_span();

	std::visit(
	    [&](auto block_dim, auto num_histograms, auto num_sh_histograms) {
		    async::fill(stream, tickets, 0);
		    auto num_warps = block_dim / warp_size;
		    gsl_Expects(num_warps % num_sh_histograms == 0);
		    gsl_Expects(num_warps >= num_sh_histograms);
		    auto c =
		        cuda::make_launch_config(int(num_histograms), int(block_dim));

		    auto tmp_mem = make_not_a_vector<int>(
		        num_histograms * histogram_length, delayed_memory_resource);

		    auto histograms = tmp_mem.to_span();

		    cuda::enqueue_launch(
		        kernel::bin_values256_threadfence<T,
		                                          block_dim,
		                                          num_histograms,
		                                          num_sh_histograms,
		                                          F0,
		                                          F1>,
		        stream,
		        c,
		        values.data(),
		        values.size(),
		        histograms.data(),
		        bit_offset,
		        prefix,
		        unary_functor,
		        bin_index_transform,
		        tickets.data(),
		        tickets.data() + 1);
	    },
	    block_dim_v,
	    num_histograms_v,
	    num_sh_histograms_v);
}

template <typename T, class MemoryResource, class F0, class F1>
void bin_values256_atomic(
    cuda::stream_t& stream,
    gsl_lite::span<const T> values,
    gsl_lite::span<int> histogram,
    int bit_offset,
    uint64_t prefix,
    int block_dim,
    int grid_dim,
    int num_sh_histograms,
    F0 unary_functor,
    F1 bin_index_transform,
    [[maybe_unused]] MemoryResource& delayed_memory_resource) {

	constexpr int histogram_length = 256;
	gsl_Expects(histogram.size() == histogram_length);

	auto block_dim_v =
	    makeshift::expand(block_dim, MAKESHIFT_CONSTVAL(std::array{256, 512}));
	auto grid_dim_v = makeshift::expand(
	    grid_dim,
	    MAKESHIFT_CONSTVAL(std::array{68, 2 * 68, 3 * 68, 4 * 68, 8 * 68}));
	auto num_sh_histograms_v = makeshift::expand(
	    num_sh_histograms, MAKESHIFT_CONSTVAL(std::array{1, 2, 3, 4, 8}));

	std::visit(
	    [&](auto block_dim, auto grid_dim, auto num_sh_histograms) {
		    async::fill(stream, histogram, 0);
		    auto num_warps = block_dim / warp_size;
		    gsl_Expects(num_warps % num_sh_histograms == 0);
		    gsl_Expects(num_warps >= num_sh_histograms);
		    auto c = cuda::make_launch_config(int(grid_dim), int(block_dim));

		    cuda::enqueue_launch(kernel::bin_values256_atomic<T,
		                                                      block_dim,
		                                                      grid_dim,
		                                                      num_sh_histograms,
		                                                      F0,
		                                                      F1>,
		                         stream,
		                         c,
		                         values.data(),
		                         values.size(),
		                         histogram.data(),
		                         bit_offset,
		                         prefix,
		                         unary_functor,
		                         bin_index_transform);
	    },
	    block_dim_v,
	    grid_dim_v,
	    num_sh_histograms_v);
}

template <typename T, class MemoryResource, class F0, class F1>
void bin_values256_atomic_with_ptr(
    cuda::stream_t& stream,
    gsl_lite::span<const T> values,
    gsl_lite::span<int> histogram,
    int bit_offset,
    uint64_t* prefix,
    int* k,
    int k0,
    bool use_k0_and_zero_prefix,
    int block_dim,
    int grid_dim,
    int num_sh_histograms,
    F0 unary_functor,
    F1 bin_index_transform,
    [[maybe_unused]] MemoryResource& delayed_memory_resource) {

	constexpr int histogram_length = 256;
	gsl_Expects(histogram.size() == histogram_length);

	auto block_dim_v =
	    makeshift::expand(block_dim, MAKESHIFT_CONSTVAL(std::array{256, 512}));
	auto grid_dim_v = makeshift::expand(
	    grid_dim,
	    MAKESHIFT_CONSTVAL(std::array{68, 2 * 68, 3 * 68, 4 * 68, 8 * 68}));
	auto num_sh_histograms_v = makeshift::expand(
	    num_sh_histograms, MAKESHIFT_CONSTVAL(std::array{1, 2, 3, 4, 8}));
	auto use_k0_and_zero_prefix_v = makeshift::expand(
	    use_k0_and_zero_prefix, MAKESHIFT_CONSTVAL(std::array{0, 1}));

	std::visit(
	    [&](auto block_dim,
	        auto grid_dim,
	        auto num_sh_histograms,
	        auto use_k0_and_zero_prefix) {
		    async::fill(stream, histogram, 0);
		    auto num_warps = block_dim / warp_size;
		    gsl_Expects(num_warps % num_sh_histograms == 0);
		    gsl_Expects(num_warps >= num_sh_histograms);
		    auto c = cuda::make_launch_config(int(grid_dim), int(block_dim));

		    cuda::enqueue_launch(
		        kernel::bin_values256_atomic_with_ptr<T,
		                                              block_dim,
		                                              grid_dim,
		                                              num_sh_histograms,
		                                              use_k0_and_zero_prefix,
		                                              F0,
		                                              F1>,
		        stream,
		        c,
		        values.data(),
		        values.size(),
		        histogram.data(),
		        bit_offset,
		        prefix,
		        k,
		        k0,
		        unary_functor,
		        bin_index_transform);
	    },
	    block_dim_v,
	    grid_dim_v,
	    num_sh_histograms_v,
	    use_k0_and_zero_prefix_v);
}

} // namespace async

template <typename T, class MemoryResource>
std::tuple<uint64_t, int> k_largest_values_abs_radix(
    cuda::stream_t& stream,
    gsl_lite::span<const T> values,
    int k,
    MemoryResource& delayed_memory_resource) {

	auto unary_functor = [] __host__ __device__(T x) {
		using std::abs;
		return abs(x);
	};
	using IH = int;
	auto bin_index_transform = [] __host__ __device__(IH i) { return i; };

	constexpr int histogram_length = 256;
	auto tmp = make_not_a_vector<IH>(histogram_length, delayed_memory_resource);

	auto histogram = tmp.to_span();

	// {
	// 	const size_t s = histogram.size() * sizeof(IH);
	// 	void* ptr = reinterpret_cast<void*>(histogram.data());
	// 	cuda::memory::managed::region_t mr{ptr, s};
	// 	static auto device = stream.device();
	// 	static bool b = false;
	// 	if (!b) {
	// 		mr.set_preferred_location(device);
	// 		b = true;
	// 	}
	// }

	uint64_t prefix = 0;
	for (int bit_offset = 0; bit_offset < int(sizeof(T) * 8); bit_offset += 8) {
		async::bin_values256<T>(
		    stream,
		    values,
		    histogram,
		    bit_offset,
		    prefix,
		    256, // block dim // params determined empirically by benchmarks
		    3 * 68, // num histos
		    1, // num sh histo
		    unary_functor,
		    bin_index_transform,
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

template <typename T, class MemoryResource>
std::tuple<uint64_t, int> k_largest_values_abs_radix_atomic(
    cuda::stream_t& stream,
    gsl_lite::span<const T> values,
    int k,
    MemoryResource& delayed_memory_resource) {

	auto unary_functor = [] __host__ __device__(T x) {
		using std::abs;
		return abs(x);
	};
	using IH = int;
	auto bin_index_transform = [] __host__ __device__(IH i) { return i; };

	constexpr int histogram_length = 256;
	auto tmp = make_not_a_vector<IH>(histogram_length, delayed_memory_resource);

	auto histogram = tmp.to_span();

	//{
	//	const size_t s = histogram.size() * sizeof(IH);
	//	void* ptr = reinterpret_cast<void*>(histogram.data());
	//	cuda::memory::managed::region_t mr{ptr, s};
	//	static auto device = stream.device();
	//	static bool b = false;
	//	if (!b) {
	//		//mr.set_preferred_location(device);
	//		//cudaMemAdvise(ptr, s, cudaMemAdviseSetAccessedBy, cudaCpuDeviceId);
	//		cudaMemAdvise(ptr, s, cudaMemAdviseSetAccessedBy, device.id());
	//		b = true;
	//	}
	//}

	uint64_t prefix = 0;
	for (int bit_offset = 0; bit_offset < int(sizeof(T) * 8); bit_offset += 8) {
		async::bin_values256_atomic<T>(
		    stream,
		    values,
		    histogram,
		    bit_offset,
		    prefix,
		    256, // block dim // params determined empirically by benchmarks
		    68 * 4, // grid dim
		    1, // num sh histo
		    unary_functor,
		    bin_index_transform,
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

	// HACK BEGIN
	// constexpr std::array<uint64_t, 4> prefixes{0, 75, 19224, 4921493};
	// for (int bit_offset = 0, i = 0; bit_offset < int(sizeof(T) * 8);
	//      bit_offset += 8, ++i) {
	// 	uint64_t prefix = prefixes[i];
	// 	async::bin_values256_atomic<T>(
	// 	    stream,
	// 	    values,
	// 	    histogram,
	// 	    bit_offset,
	// 	    prefix,
	// 	    256, // block dim // params determined empirically by benchmarks
	// 	    68 * 4, // grid dim
	// 	    1, // num sh histo
	// 	    unary_functor,
	// 	    bin_index_transform,
	// 	    delayed_memory_resource);
	// 	stream.synchronize();
	// 	// int acc = histogram[histogram_length - 1];
	// 	// int acc_prev = 0;
	// 	// for (int i = histogram_length - 2; i >= 0; --i) {
	// 	// 	acc += histogram[i];
	// 	// 	if (acc >= k) {
	// 	// 		prefix = (prefix << 8) | uint64_t(i);
	// 	// 		if (acc == k) {
	// 	// 			// all values with this prefix and larger are included
	// 	// 			return {prefix, bit_offset + 8};
	// 	// 		}
	// 	// 		k = k - acc_prev;
	// 	// 		break;
	// 	// 	}
	// 	// 	acc_prev = acc;
	// 	// }
	// }
	// return {prefixes[3], sizeof(T) * 8};
	// HACK END
}

template <typename T, class MemoryResource>
std::tuple<uint64_t, int> k_largest_values_abs_radix_atomic_devicehisto(
    cuda::stream_t& stream,
    gsl_lite::span<const T> values,
    int k,
    MemoryResource& delayed_memory_resource) {

	auto unary_functor = [] __host__ __device__(T x) {
		using std::abs;
		return abs(x);
	};
	using IH = int;

	constexpr int histogram_length = 256;
	auto bin_index_transform = [] __host__ __device__(IH i) {
		return histogram_length - i - 1;
	};

	auto tmp = make_not_a_vector<IH>(histogram_length, delayed_memory_resource);
	auto tmp0 = make_not_a_vector<uint64_t>(1, delayed_memory_resource);
	auto tmp1 = make_not_a_vector<int>(1, delayed_memory_resource);
	auto prefix_s = tmp0.to_span();
	auto k_s = tmp1.to_span();
	async::fill(stream, k_s, k);

	auto histogram = tmp.to_span();
	uint64_t prefix = 0;

	for (int bit_offset = 0; bit_offset < int(sizeof(T) * 8); bit_offset += 8) {
		async::bin_values256_atomic<T>(
		    stream,
		    values,
		    histogram,
		    bit_offset,
		    prefix,
		    256, // block dim // params determined empirically by benchmarks
		    68 * 4, // grid dim
		    1, // num sh histo
		    unary_functor,
		    bin_index_transform,
		    delayed_memory_resource);

		cuda::enqueue_launch(kernel::k_select_radix_from_histogram<IH, 256>,
		                     stream,
		                     cuda::make_launch_config(1, 256),
		                     histogram.data(),
		                     bit_offset,
		                     prefix_s.data(),
		                     k_s.data());
		stream.synchronize();
		prefix = (prefix << 8) | prefix_s[0];
		k = k_s[0];
		if (k <= 0) {
			return {prefix, bit_offset + 8};
		}
	}
	return {prefix, sizeof(T) * 8};
	// HACK BEGIN
	// constexpr std::array<uint64_t, 4> prefixes{0, 75, 19224, 4921493};
	// for (int bit_offset = 0, i = 0; bit_offset < int(sizeof(T) * 8);
	//      bit_offset += 8, ++i) {
	// 	uint64_t prefix = prefixes[i];
	// 	async::bin_values256_atomic<T>(
	// 	    stream,
	// 	    values,
	// 	    histogram,
	// 	    bit_offset,
	// 	    prefix,
	// 	    256, // block dim // params determined empirically by benchmarks
	// 	    68 * 4, // grid dim
	// 	    1, // num sh histo
	// 	    unary_functor,
	// 	    bin_index_transform,
	// 	    delayed_memory_resource);

	// 	cuda::enqueue_launch(kernel::k_select_radix_from_histogram<IH, 256>,
	// 	                     stream,
	// 	                     cuda::make_launch_config(1, 256),
	// 	                     histogram.data(),
	// 	                     bit_offset,
	// 	                     prefix_s.data(),
	// 	                     k_s.data());
	// 	// stream.synchronize();
	// 	// prefix = (prefix << 8) | prefix_s[0];
	// 	// k = k_s[0];
	// 	// if (k <= 0) {
	// 	// 	return {prefix, bit_offset + 8};
	// 	// }
	// }
	// return {prefixes[3], sizeof(T) * 8};
	// HACK END
}

namespace async {

template <typename T, class MemoryResource>
void k_largest_values_abs_radix_atomic_devicehisto_with_ptr(
    cuda::stream_t& stream,
    gsl_lite::span<const T> values,
    uint64_t* prefix, // only output
    int* bit_offset, //only output
    int k,
    MemoryResource& delayed_memory_resource) {

	auto unary_functor = [] __host__ __device__(T x) {
		using std::abs;
		return abs(x);
	};
	using IH = int;

	constexpr int histogram_length = 256;
	auto bin_index_transform = [] __host__ __device__(IH i) {
		return histogram_length - i - 1;
	};

	auto tmp = make_not_a_vector<IH>(histogram_length, delayed_memory_resource);
	auto tmp1 = make_not_a_vector<int>(1, delayed_memory_resource);
	auto k_s = tmp1.to_span();
	gsl_lite::span<int> bit_offset_s({bit_offset, 1});
	gsl_lite::span<uint64_t> prefix_s({prefix, 1});

	auto histogram = tmp.to_span();

	for (int bit_offset = 0; bit_offset < int(sizeof(T) * 8); bit_offset += 8) {

		async::bin_values256_atomic_with_ptr<T>(
		    stream,
		    values,
		    histogram,
		    bit_offset,
		    prefix_s.data(),
		    k_s.data(),
		    k,
		    bit_offset == 0,
		    256, // block dim // params determined empirically by benchmarks
		    68 * 4, // grid dim
		    1, // num sh histo
		    unary_functor,
		    bin_index_transform,
		    delayed_memory_resource);

		auto flag_v = makeshift::expand(bit_offset == 0,
		                                MAKESHIFT_CONSTVAL(std::array{0, 1}));
		std::visit(
		    [&](auto flag) {
			    cuda::enqueue_launch(
			        kernel::
			            k_select_radix_from_histogram_with_ptr<IH, 256, flag>,
			        stream,
			        cuda::make_launch_config(1, 256),
			        histogram.data(),
			        bit_offset_s.data(),
			        prefix_s.data(),
			        k_s.data(),
			        k);
		    },
		    flag_v);
	}
}

} // namespace async

template <typename T, class MemoryResource>
std::tuple<uint64_t, int> k_largest_values_abs_radix_with_cub(
    cuda::stream_t& stream,
    gsl_lite::span<const T> values,
    int k,
    MemoryResource& delayed_memory_resource) {

	constexpr int histogram_length = 256;
	auto tmp =
	    make_not_a_vector<int>(histogram_length, delayed_memory_resource);
	auto histogram = tmp.to_span();

	using I = typename thrustshift::make_uintegral_of_equal_size<T>::type;

	uint64_t prefix = 0;
	I lower_level = 0; // inclusive
	I upper_level = std::numeric_limits<I>::max(); // exclusive
	const int N = values.size();

	for (int bit_offset = 0; bit_offset < int(sizeof(T) * 8); bit_offset += 8) {

		// __host__ needed only for return type, from CUDA 12 on one could
		// use cuda::proclaim_return_type instead
		auto sample_iterator = thrust::make_transform_iterator(
		    values.data(),
		    [prefix, bit_offset] __host__ __device__(const T& x) {
			    using std::abs;
			    const T abs_x = abs(x);
			    const I i = *reinterpret_cast<I*>((void*) (&abs_x));
			    return i;
		    });
		async::bin_values_into_histogram(stream,
		                                 sample_iterator,
		                                 histogram.begin(),
		                                 histogram_length,
		                                 lower_level,
		                                 upper_level,
		                                 N,
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
		lower_level = prefix << (sizeof(I) * 8 - (bit_offset + 8));
		upper_level = (prefix + 1) << (sizeof(I) * 8 - (bit_offset + 8));
	}
	return {prefix, sizeof(T) * 8};
}

// selected_values.size() == values.size() because CUB might select more values, if e.g. values are all equal
template <typename T, class MemoryResource>
void select_k_largest_values_abs(cuda::stream_t& stream,
                                 gsl_lite::span<const T> values,
                                 gsl_lite::span<T> selected_values,
                                 gsl_lite::span<int> selected_indices,
                                 int k,
                                 MemoryResource& delayed_memory_resource) {

	const auto tup = thrustshift::k_largest_values_abs_radix<T>(
	    stream, values, k, delayed_memory_resource);
	const auto prefix = std::get<0>(tup);
	const auto bit_offset = std::get<1>(tup);
	auto select_op = [prefix, bit_offset] __host__ __device__(
	                     const thrust::tuple<T, int>& tup) {
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
	                            selected_indices,
	                            num_selected.data(),
	                            select_op,
	                            delayed_memory_resource);
	stream.synchronize();
}

template <typename T, class MemoryResource>
void select_k_largest_values_abs_atomic(
    cuda::stream_t& stream,
    gsl_lite::span<const T> values,
    gsl_lite::span<T> selected_values,
    gsl_lite::span<int> selected_indices,
    int k,
    MemoryResource& delayed_memory_resource) {

	const auto tup = thrustshift::k_largest_values_abs_radix_atomic<T>(
	    stream, values, k, delayed_memory_resource);
	const auto prefix = std::get<0>(tup);
	const auto bit_offset = std::get<1>(tup);
	auto select_op = [prefix, bit_offset] __host__ __device__(
	                     const thrust::tuple<T, int>& tup) {
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
	                            selected_indices,
	                            num_selected.data(),
	                            select_op,
	                            delayed_memory_resource);
	stream.synchronize();
}

// selected_values.size() == values.size() because CUB might select more values, if e.g. values are all equal
template <typename T, class MemoryResource>
void select_k_largest_values_abs_with_cub(
    cuda::stream_t& stream,
    gsl_lite::span<const T> values,
    gsl_lite::span<T> selected_values,
    gsl_lite::span<int> selected_indices,
    int k,
    MemoryResource& delayed_memory_resource) {

	const auto tup = thrustshift::k_largest_values_abs_radix_with_cub<T>(
	    stream, values, k, delayed_memory_resource);
	const auto prefix = std::get<0>(tup);
	const auto bit_offset = std::get<1>(tup);
	auto select_op = [prefix, bit_offset] __host__ __device__(
	                     const thrust::tuple<T, int>& tup) {
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
	                            selected_indices,
	                            num_selected.data(),
	                            select_op,
	                            delayed_memory_resource);
	stream.synchronize();
}

namespace cooperative {

template <typename T, class MemoryResource>
void select_k_largest_values_abs(cuda::stream_t& stream,
                                 gsl_lite::span<const T> values,
                                 gsl_lite::span<T> selected_values,
                                 gsl_lite::span<int> selected_indices,
                                 int k,
                                 MemoryResource& delayed_memory_resource) {

	constexpr int histogram_length = 256;
	constexpr int block_dim = 256;
	constexpr int grid_dim = 2 * 68;
	constexpr int num_sh_histograms = 1;

	auto tmp = make_not_a_vector<int>(3, delayed_memory_resource);
	auto tmp1 = make_not_a_vector<uint64_t>(1, delayed_memory_resource);
	auto tmp2 = make_not_a_vector<unsigned>(histogram_length * grid_dim,
	                                        delayed_memory_resource);

	auto num_selected = tmp.to_span().first(1);
	auto bit_offset_s = tmp.to_span().subspan(1, 1);
	auto prefix_s = tmp1.to_span().first(1);
	auto histograms = tmp2.to_span();
	auto k_s = tmp.to_span().subspan(2, 1);
	k_s[0] = k;

	auto c = cuda::make_launch_config(grid_dim, block_dim);
	c.block_cooperation = true;

	const int N = values.size();

	cuda::enqueue_launch(
	    kernel::
	        k_select_radix<T, unsigned, block_dim, grid_dim, num_sh_histograms>,
	    stream,
	    c,
	    values.data(),
	    N,
	    histograms.data(),
	    bit_offset_s.data(),
	    prefix_s.data(),
	    k_s.data());
	stream.synchronize();

	const auto prefix = prefix_s[0];
	const auto bit_offset = bit_offset_s[0];
	auto select_op = [prefix, bit_offset] __host__ __device__(
	                     const thrust::tuple<T, int>& tup) {
		using std::abs;
		auto x = abs(thrust::get<0>(tup));
		using I = typename thrustshift::make_uintegral_of_equal_size<T>::type;
		const I i = *reinterpret_cast<I*>((void*) (&x));
		return (i >> sizeof(I) * 8 - bit_offset) >= static_cast<I>(prefix);
	};

	async::select_if_with_index(stream,
	                            values,
	                            selected_values,
	                            selected_indices,
	                            num_selected.data(),
	                            select_op,
	                            delayed_memory_resource);
	stream.synchronize();
}

template <typename T, class MemoryResource>
void select_k_largest_values_abs2(cuda::stream_t& stream,
                                  gsl_lite::span<const T> values,
                                  gsl_lite::span<T> selected_values,
                                  gsl_lite::span<int> selected_indices,
                                  int k,
                                  MemoryResource& delayed_memory_resource) {

	constexpr int histogram_length = 256;
	constexpr int block_dim = 256;
	constexpr int grid_dim = 2 * 68;
	constexpr int num_sh_histograms = 1;

	auto tmp = make_not_a_vector<int>(3, delayed_memory_resource);
	auto tmp1 = make_not_a_vector<uint64_t>(1, delayed_memory_resource);
	auto tmp2 = make_not_a_vector<unsigned>(histogram_length * grid_dim * 4,
	                                        delayed_memory_resource);

	auto num_selected = tmp.to_span().first(1);
	auto bit_offset_s = tmp.to_span().subspan(1, 1);
	auto prefix_s = tmp1.to_span().first(1);
	auto histograms = tmp2.to_span();
	auto k_s = tmp.to_span().subspan(2, 1);
	k_s[0] = k;

	auto c = cuda::make_launch_config(grid_dim, block_dim);
	c.block_cooperation = true;
	const int N = values.size();

	cuda::enqueue_launch(kernel::k_select_radix2<T,
	                                             unsigned,
	                                             block_dim,
	                                             grid_dim,
	                                             num_sh_histograms>,
	                     stream,
	                     c,
	                     values.data(),
	                     N,
	                     histograms.data(),
	                     bit_offset_s.data(),
	                     prefix_s.data(),
	                     k);
	stream.synchronize();

	const auto prefix = prefix_s[0];
	const auto bit_offset = bit_offset_s[0];
	auto select_op = [prefix, bit_offset] __host__ __device__(
	                     const thrust::tuple<T, int>& tup) {
		using std::abs;
		auto x = abs(thrust::get<0>(tup));
		using I = typename thrustshift::make_uintegral_of_equal_size<T>::type;
		const I i = *reinterpret_cast<I*>((void*) (&x));
		return (i >> sizeof(I) * 8 - bit_offset) >= static_cast<I>(prefix);
	};

	async::select_if_with_index(stream,
	                            values,
	                            selected_values,
	                            selected_indices,
	                            num_selected.data(),
	                            select_op,
	                            delayed_memory_resource);
	stream.synchronize();
}

} // namespace cooperative

namespace dynamic_parallelism {

template <typename T, class MemoryResource>
std::tuple<uint64_t, int> k_largest_values_abs_radix_atomic_binning(
    cuda::stream_t& stream,
    gsl_lite::span<const T> values,
    int k,
    bool nosync,
    MemoryResource& delayed_memory_resource) {

	constexpr int histogram_length = 256;
	constexpr int block_dim = 256;
	constexpr int child_grid_dim = 68 * 4;
	constexpr int num_sh_histograms = 1;

	using IH = int;
	auto tmp = make_not_a_vector<int>(1, delayed_memory_resource);
	auto tmp1 = make_not_a_vector<uint64_t>(1, delayed_memory_resource);
	auto tmp2 =
	    make_not_a_vector<IH>(histogram_length, delayed_memory_resource);

	auto bit_offset_s = tmp.to_span();
	auto prefix_s = tmp1.to_span();
	auto histogram = tmp2.to_span();

	auto c = cuda::make_launch_config(1, block_dim);
	const int N = values.size();

	cuda::enqueue_launch(
	    kernel::k_select_radix_dynamic_parallelism_atomic_binning<
	        T,
	        IH,
	        block_dim,
	        num_sh_histograms,
	        child_grid_dim>,
	    stream,
	    c,
	    values.data(),
	    N,
	    histogram.data(),
	    bit_offset_s.data(),
	    prefix_s.data(),
	    k);
	// Alternatively give CUB select if a lambda, which accesses the values, which are already on the GPU
	if (!nosync) {
		stream.synchronize();
		return {prefix_s[0], bit_offset_s[0]};
	}
	else {
		return {0, 0};
	}
}

template <typename T, class MemoryResource>
void select_k_largest_values_abs(cuda::stream_t& stream,
                                 gsl_lite::span<const T> values,
                                 gsl_lite::span<T> selected_values,
                                 gsl_lite::span<int> selected_indices,
                                 int k,
                                 MemoryResource& delayed_memory_resource) {

	constexpr int histogram_length = 256;
	constexpr int block_dim = 256;
	constexpr int num_sh_histograms = 1;
	constexpr int num_gl_histograms = 1 * 68;

	using IH = int;
	auto tmp = make_not_a_vector<int>(3, delayed_memory_resource);
	auto tmp1 = make_not_a_vector<uint64_t>(1, delayed_memory_resource);
	auto tmp2 = make_not_a_vector<IH>(histogram_length * num_gl_histograms,
	                                  delayed_memory_resource);

	auto num_selected = tmp.to_span().first(1);
	auto bit_offset_s = tmp.to_span().subspan(1, 1);
	auto prefix_s = tmp1.to_span().first(1);
	auto histograms = tmp2.to_span();
	// auto k_s = tmp.to_span().subspan(2, 1);
	// k_s[0] = k;

	auto c = cuda::make_launch_config(1, block_dim);
	const int N = values.size();

	cuda::enqueue_launch(
	    kernel::k_select_radix_dynamic_parallelism<T,
	                                               IH,
	                                               block_dim,
	                                               num_sh_histograms,
	                                               num_gl_histograms>,
	    stream,
	    c,
	    values.data(),
	    N,
	    histograms.data(),
	    bit_offset_s.data(),
	    prefix_s.data(),
	    k);
	stream.synchronize();

	const auto prefix = prefix_s[0];
	const auto bit_offset = bit_offset_s[0];
	auto select_op = [prefix, bit_offset] __host__ __device__(
	                     const thrust::tuple<T, int>& tup) {
		using std::abs;
		auto x = abs(thrust::get<0>(tup));
		using I = typename thrustshift::make_uintegral_of_equal_size<T>::type;
		const I i = *reinterpret_cast<I*>((void*) (&x));
		return (i >> sizeof(I) * 8 - bit_offset) >= static_cast<I>(prefix);
	};

	async::select_if_with_index(stream,
	                            values,
	                            selected_values,
	                            selected_indices,
	                            num_selected.data(),
	                            select_op,
	                            delayed_memory_resource);
	stream.synchronize();
}

} // namespace dynamic_parallelism

namespace async {

template <typename T, class MemoryResource>
void select_k_largest_values_abs(cuda::stream_t& stream,
                                 gsl_lite::span<const T> values,
                                 gsl_lite::span<T> selected_values,
                                 gsl_lite::span<int> selected_indices,
                                 int k,
                                 MemoryResource& delayed_memory_resource) {

	auto tmp0 = make_not_a_vector<uint64_t>(1, delayed_memory_resource);
	auto prefix_s = tmp0.to_span();
	auto tmp1 = make_not_a_vector<int>(1, delayed_memory_resource);
	auto bit_offset_s = tmp1.to_span();

	uint64_t* prefix_ptr = prefix_s.data();
	int* bit_offset_ptr = bit_offset_s.data();

	async::k_largest_values_abs_radix_atomic_devicehisto_with_ptr<T>(
	    stream, values, prefix_ptr, bit_offset_ptr, k, delayed_memory_resource);

	auto select_op = [prefix_ptr, bit_offset_ptr] __host__ __device__(
	                     const thrust::tuple<T, int>& tup) {
		using std::abs;
		auto x = abs(thrust::get<0>(tup));
		using I = typename thrustshift::make_uintegral_of_equal_size<T>::type;
		const I i = *reinterpret_cast<I*>((void*) (&x));
		// No performance effects measured when we load these values here from
		// global memory. Probably they end up in the caches and can be loaded fast.
		uint64_t prefix = *prefix_ptr;
		int bit_offset = *bit_offset_ptr;
		// uint64_t prefix = 1259902258;
		// int bit_offset = 32;
		return (i >> sizeof(I) * 8 - bit_offset) >= static_cast<I>(prefix);
	};
	auto tmp = make_not_a_vector<int>(1, delayed_memory_resource);
	auto num_selected = tmp.to_span();

	async::select_if_with_index(stream,
	                            values,
	                            selected_values,
	                            selected_indices,
	                            num_selected.data(),
	                            select_op,
	                            delayed_memory_resource);
}

} // namespace async

} // namespace thrustshift
