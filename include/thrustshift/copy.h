#pragma once

#include <type_traits>

#include <gsl-lite/gsl-lite.hpp>

#include <thrustshift/defines.h>
#include <thrustshift/fill.h>

namespace thrustshift {

namespace kernel {

template <typename SrcT, typename DstT>
__global__ void copy(gsl_lite::span<const SrcT> src, gsl_lite::span<DstT> dst) {

	const auto gtid = threadIdx.x + blockIdx.x * blockDim.x;
	if (gtid < src.size()) {
		dst[gtid] = src[gtid];
	}
}

template <typename SrcT, typename DstT, typename T, typename I>
__global__ void copy_find(gsl_lite::span<const SrcT> src,
                          gsl_lite::span<DstT> dst,
                          T value,
                          I* pos) {

	const auto gtid = threadIdx.x + blockIdx.x * blockDim.x;
	if (gtid < src.size()) {
		const auto src_value = src[gtid];
		dst[gtid] = src_value;
		if (src_value == value) {
			*pos = gtid;
		}
	}
}

} // namespace kernel

namespace async {

//! thrust uses sometimes a cudaMemcpyAsync instead of a copy kernel
template <class SrcRange, class DstRange>
void copy(cudaStream_t& stream, SrcRange&& src, DstRange&& dst) {
	gsl_Expects(src.size() == dst.size());

	if (src.empty()) {
		return;
	}

	using src_value_type =
	    typename std::remove_reference<SrcRange>::type::value_type;
	using dst_value_type =
	    typename std::remove_reference<DstRange>::type::value_type;

	constexpr unsigned block_dim = 128;
	const unsigned grid_dim = (src.size() + block_dim - 1) / block_dim;
	kernel::copy<src_value_type, dst_value_type>
	    <<<grid_dim, block_dim, 0, stream>>>(src, dst);
	THRUSTSHIFT_CHECK_CUDA_ERROR(cudaGetLastError());
}

//! Copy and search for an element. If the element occurs more than once, it
//! is undefined which of the valid positions is returned. If the element does
//! not occur `pos` is unchanged.
template <class SrcRange, class DstRange, typename T, typename I>
void copy_find(cudaStream_t& stream,
               SrcRange&& src,
               DstRange&& dst,
               const T& value,
               I* pos) {
	gsl_Expects(src.size() == dst.size());
	gsl_Expects(pos != nullptr);

	if (src.empty()) {
		return;
	}

	using src_value_type =
	    typename std::remove_reference<SrcRange>::type::value_type;
	using dst_value_type =
	    typename std::remove_reference<DstRange>::type::value_type;

	constexpr unsigned block_dim = 128;
	const unsigned grid_dim = (src.size() + block_dim - 1) / block_dim;
	kernel::copy_find<src_value_type, dst_value_type, T, I>
	    <<<grid_dim, block_dim, 0, stream>>>(src, dst, value, pos);
	THRUSTSHIFT_CHECK_CUDA_ERROR(cudaGetLastError());
}

} // namespace async

namespace detail {

template <int BLOCK_DIM,
          int NUM_ELEMENTS,
          int LD_FIRST = NUM_ELEMENTS,
          int LD_RESULT = NUM_ELEMENTS,
          int NUM_PER_ROW_FIRST = NUM_ELEMENTS,
          int NUM_PER_ROW_RESULT = NUM_ELEMENTS>
struct helper_t {
	template <class iteratorA_t, class iteratorB_t, class F>
	THRUSTSHIFT_FHD static void block_copy_even(iteratorA_t first,
	                                            iteratorB_t result,
	                                            int tid,
	                                            F f) {
		if (BLOCK_DIM < NUM_ELEMENTS) {
// pragma unroll is device code specific
#ifdef __CUDA_ARCH__
#pragma unroll
#endif
			for (int i = 0; i < NUM_ELEMENTS - BLOCK_DIM; i += BLOCK_DIM) {
				f(first,
				  result,
				  i +
				      ((i + tid) / NUM_PER_ROW_FIRST) *
				          (LD_FIRST - NUM_PER_ROW_FIRST) +
				      tid,
				  i +
				      ((i + tid) / NUM_PER_ROW_RESULT) *
				          (LD_RESULT - NUM_PER_ROW_RESULT) +
				      tid);
			}
		}
	}

	template <class iteratorA_t, class iteratorB_t, class F>
	THRUSTSHIFT_FHD static void block_copy_tail(iteratorA_t first,
	                                            iteratorB_t result,
	                                            int tid,
	                                            F f) {
		if (NUM_ELEMENTS % BLOCK_DIM == 0) {
			f(first,
			  result,
			  (NUM_ELEMENTS - BLOCK_DIM) +
			      (((NUM_ELEMENTS - BLOCK_DIM) + tid) / NUM_PER_ROW_FIRST) *
			          (LD_FIRST - NUM_PER_ROW_FIRST) +
			      tid,
			  (NUM_ELEMENTS - BLOCK_DIM) +
			      (((NUM_ELEMENTS - BLOCK_DIM) + tid) / NUM_PER_ROW_RESULT) *
			          (LD_RESULT - NUM_PER_ROW_RESULT) +
			      tid);
		}
		else if (tid + (NUM_ELEMENTS / BLOCK_DIM) * BLOCK_DIM < NUM_ELEMENTS) {
			constexpr int j = (NUM_ELEMENTS / BLOCK_DIM) * BLOCK_DIM;
			f(first,
			  result,
			  j +
			      ((j + tid) / NUM_PER_ROW_FIRST) *
			          (LD_FIRST - NUM_PER_ROW_FIRST) +
			      tid,
			  j +
			      ((j + tid) / NUM_PER_ROW_RESULT) *
			          (LD_RESULT - NUM_PER_ROW_RESULT) +
			      tid);
		}
	}
}; // helper

template <int BLOCK_DIM, int NUM_ELEMENTS>
struct helper_t<BLOCK_DIM,
                NUM_ELEMENTS,
                NUM_ELEMENTS,
                NUM_ELEMENTS,
                NUM_ELEMENTS,
                NUM_ELEMENTS> {
	template <class iteratorA_t, class iteratorB_t, class F>
	THRUSTSHIFT_FHD static void block_copy_even(iteratorA_t first,
	                                            iteratorB_t result,
	                                            int tid,
	                                            F f) {
		if (BLOCK_DIM < NUM_ELEMENTS) {
// pragma unroll is device code specific
#ifdef __CUDA_ARCH__
#pragma unroll
#endif
			for (int i = 0; i < NUM_ELEMENTS - BLOCK_DIM; i += BLOCK_DIM) {
				const int j = i + tid;
				f(first, result, j, j);
			}
		}
	}

	template <class iteratorA_t, class iteratorB_t, class F>
	THRUSTSHIFT_FHD static void block_copy_tail(iteratorA_t first,
	                                            iteratorB_t result,
	                                            int tid,
	                                            F f) {
		if (NUM_ELEMENTS % BLOCK_DIM == 0) {
			const int j = NUM_ELEMENTS - BLOCK_DIM + tid;
			f(first, result, j, j);
		}
		else if (tid + (NUM_ELEMENTS / BLOCK_DIM) * BLOCK_DIM < NUM_ELEMENTS) {
			const int j = (NUM_ELEMENTS / BLOCK_DIM) * BLOCK_DIM + tid;
			f(first, result, j, j);
		}
	}
}; // struct helper_t

//! Specialization for equal block dimension and number of elements
template <int BD_AND_NE>
struct helper_t<BD_AND_NE,
                BD_AND_NE,
                BD_AND_NE,
                BD_AND_NE,
                BD_AND_NE,
                BD_AND_NE> {
	template <class iteratorA_t, class iteratorB_t, class F>
	THRUSTSHIFT_FHD static void block_copy_even(iteratorA_t first,
	                                            iteratorB_t result,
	                                            int tid,
	                                            F f) {
	}

	template <class iteratorA_t, class iteratorB_t, class F>
	THRUSTSHIFT_FHD static void block_copy_tail(iteratorA_t first,
	                                            iteratorB_t result,
	                                            int tid,
	                                            F f) {
		f(first, result, tid, tid);
	}

}; // struct helper_t

}; // namespace detail

/* \brief Copy data blockwise with efficient loop unrolling.
 *
 * This function implements an efficient memory copy. This is in particular
 * useful if you use only a few threads to copy data, since this is limited
 * by register dependencies if trivially implemented. The algorithm is
 * implemented in two steps. First we copy elements subsequently and
 * block wise with the whole thread block:
 *
 *     N = 11, bdim = 4: x x x x | x x x x | o o o
 *     N = 12, bdim = 4: x x x x | x x x x | o o o o
 *     N = 13, bdim = 4: x x x x | x x x x | x x x x | o
 *
 * Thus the last elements are not copied (denoted by `o`).
 * Afterwards we must copy the remaining elements, which is
 * trivial if N % bdim == 0.
 *
 * \param BLOCK_DIM Current thread block dimension
 * \param NUM_ELEMENTS total amount of elements to copy
 * \param LD_RESULT Leading dimension on result. Only used in case of two dimensional
 *        arranged data. This is equal to the number of elements, which
 *        must be skipped to get an element in the next row. This is
 *        useful if you want to skip some of the columns of the 2D data:
 *
 *           |    ld     |
 *            x x x x o o
 *            x x x x o o
 *            x x x x o o
 *            x x x x o o
 *           |  nepr |
 *
 *        `nepr` = num elements per row
 *        `ld` = leading dimension
 *        If you want to copy only the `x` elements.
 *
 * \param NUM_PER_ROW Only used in case of two dimensional data copies
 * \param first iterator to first read element
 * \param result iterator to first result element
 * \param f A functor to write the data, e.g.
 *      ```
 *      auto default_f = [](iteratorA_t first,
 *                          iteratorB_t result,
 *                          int i_first,
 *                          int i_result) {
 *          result[i_result] = first[i_first];
 *      };
 *      ```
 *     `i_first` and `i_result` are the global indices on iterator `first` and `result`. With such
 *     a functor it is possible to modify the read and write process. E.g. you can read
 *     the data in a permuted way.
 *
 */
template <int BLOCK_DIM,
          int NUM_ELEMENTS,
          class iteratorA_t,
          class iteratorB_t,
          int LD_FIRST = NUM_ELEMENTS,
          int LD_RESULT = NUM_ELEMENTS,
          int NUM_PER_ROW_FIRST = NUM_ELEMENTS,
          int NUM_PER_ROW_RESULT = NUM_ELEMENTS,
          class F>
THRUSTSHIFT_FHD void block_copy(iteratorA_t first,
                                iteratorB_t result,
                                int tid,
                                F f) {

	detail::helper_t<BLOCK_DIM,
	                 NUM_ELEMENTS,
	                 LD_FIRST,
	                 LD_RESULT,
	                 NUM_PER_ROW_FIRST,
	                 NUM_PER_ROW_RESULT>::block_copy_even(first,
	                                                      result,
	                                                      tid,
	                                                      f);
	detail::helper_t<BLOCK_DIM,
	                 NUM_ELEMENTS,
	                 LD_FIRST,
	                 LD_RESULT,
	                 NUM_PER_ROW_FIRST,
	                 NUM_PER_ROW_RESULT>::block_copy_tail(first,
	                                                      result,
	                                                      tid,
	                                                      f);
}

template <int BLOCK_DIM,
          int NUM_ELEMENTS,
          int LD_FIRST = NUM_ELEMENTS,
          int LD_RESULT = NUM_ELEMENTS,
          int NUM_PER_ROW_FIRST = NUM_ELEMENTS,
          int NUM_PER_ROW_RESULT = NUM_ELEMENTS,
          class iteratorA_t,
          class iteratorB_t>
THRUSTSHIFT_FHD void block_copy(iteratorA_t first,
                                iteratorB_t result,
                                int tid = threadIdx.x) {

	auto default_f =
	    [](iteratorA_t first, iteratorB_t result, int i_first, int i_result) {
		    result[i_result] = first[i_first];
	    };

	block_copy<BLOCK_DIM,
	           NUM_ELEMENTS,
	           iteratorA_t,
	           iteratorB_t,
	           LD_FIRST,
	           LD_RESULT,
	           NUM_PER_ROW_FIRST,
	           NUM_PER_ROW_RESULT>(first, result, tid, default_f);
}

//! copy data without loops unrolled
template <class iteratorA_t, class iteratorB_t>
THRUSTSHIFT_FHD void block_copy(iteratorA_t first,
                                int num_elements,
                                iteratorB_t result,
                                int group_dim,
                                int ld_first,
                                int ld_result,
                                int num_per_row_first,
                                int num_per_row_result,
                                int tid) {
	auto f =
	    [](iteratorA_t first, iteratorB_t result, int i_first, int i_result) {
		    result[i_result] = first[i_first];
	    };
	//////////
	// HEAD //
	//////////
	if (group_dim < num_elements) {
		for (int i = 0; i < num_elements - group_dim; i += group_dim) {
			f(first,
			  result,
			  i +
			      ((i + tid) / num_per_row_first) *
			          (ld_first - num_per_row_first) +
			      tid,
			  i +
			      ((i + tid) / num_per_row_result) *
			          (ld_result - num_per_row_result) +
			      tid);
		}
	}
	//////////
	// TAIL //
	//////////
	if (num_elements % group_dim == 0) {
		f(first,
		  result,
		  (num_elements - group_dim) +
		      (((num_elements - group_dim) + tid) / num_per_row_first) *
		          (ld_first - num_per_row_first) +
		      tid,
		  (num_elements - group_dim) +
		      (((num_elements - group_dim) + tid) / num_per_row_result) *
		          (ld_result - num_per_row_result) +
		      tid);
	}
	else if (tid + (num_elements / group_dim) * group_dim < num_elements) {
		const int j = (num_elements / group_dim) * group_dim;
		f(first,
		  result,
		  j + ((j + tid) / num_per_row_first) * (ld_first - num_per_row_first) +
		      tid,
		  j +
		      ((j + tid) / num_per_row_result) *
		          (ld_result - num_per_row_result) +
		      tid);
	}
}

} // namespace thrustshift
