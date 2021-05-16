#pragma once

#include <bitset>
#include <type_traits>

#include <thrust/execution_policy.h>
#include <thrust/gather.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/scatter.h>

#include <cuda/define_specifiers.hpp>

#include <gsl-lite/gsl-lite.hpp>

#include <thrustshift/managed-vector.h>
#include <thrustshift/bit.h>

namespace thrustshift {

namespace permutation {

/*! \brief Return \f$ P_b P_a^T \f$
 *
 *   \param result `result[i]` is the index of element \f$ a_i \f$
 *     after permutation \f$ P_b \f$ is applied to the natural order.
 */
template <class PermutationA, class PermutationB, class ResultPermutation>
void multiply(PermutationA&& a, PermutationB&& b, ResultPermutation&& result) {

	gsl_Expects(a.size() == b.size());
	gsl_Expects(result.size() == a.size());

	using ResultIndex =
	    typename std::remove_reference<ResultPermutation>::type::value_type;

	managed_vector<ResultIndex> tmp(result.size());

	auto cit_begin = thrust::make_counting_iterator(ResultIndex(0));
	auto cit_end = thrust::make_counting_iterator(
	    gsl_lite::narrow<ResultIndex>(result.size()));

	// Rather use `.data()` in case the iterator class is not defined
	// in device code
	thrust::scatter(thrust::device, cit_begin, cit_end, b.data(), tmp.data());
	thrust::gather(thrust::device,
	               a.data(),
	               a.data() + a.size(),
	               tmp.data(),
	               result.data());
}

/*! \brief Multiply successive permutations
 *
 *  Assume you have \f$N\f$ permutations \f$P_{N-1},...,P_0\f$
 *  and \f$N\f$ linear maps \f$M_{N-1},...,M_0\f$ where each
 *  linear map \f$M_i\f$ operates on the new order described by
 *  \f$P_i\f$ relatively to the natural order. If every
 *  linear map is applied to a vector \f$x\f$ in natural order
 *  it yields to:
 *
 *  \f[
 *      P_{N-1}^T M_{N-1} P_{N-1} P_{N-2}^T M^{N-2} ... P_0^T M_0 P_0 x
 *  \f]
 *
 *  The \f$P_i\f$ permutation matrices are given in vectorial form \f$p_i\f$
 *  such that \f$(P_i)_mn = 1\f$ if \f$(p_i)_m = n\f$.
 *
 *  \param input_permutations Range of size N with ranges of length L each
 *  \param merged_permutations Range of size N - 1 with ranges of length L each.
 *    merged_permutations[i] = \f$P_{i+1} P_{i}^T\f$
 */
template <class InputPermutations, class MergedPermutations>
void multiply_successive(InputPermutations&& input_permutations,
                         MergedPermutations&& multiplied_permutations) {

	const auto N = input_permutations.size();

	if (N == 0) {
		return;
	}

	gsl_Expects(N > 0);
	gsl_Expects(N == multiplied_permutations.size() + 1);

	for (size_t i = 0; i < input_permutations.size() - 1; ++i) {
		multiply(input_permutations[i],
		         input_permutations[i + 1],
		         multiplied_permutations[i]);
	}
}

/*! \brief Bitoptimized successive pairwise permutation
 *
 *  This class saves a permutation of `i = {0,...,N-1}`, which
 *  is created by successive pairwise swaps of the sequence `i`:
 *
 *  ```cpp
 *  for (int j = 1; j < N; ++j) {
 *      if (do_swap[j]) {
 *          swap(i[j], i[j - 1]);
 *      }
 *  }
 *  ```
 *
 *  The array `do_swap[]` saves if two values are swapped and is saved in
 *  a bit pattern within this class.
 *  This can also be seen as a path in a binary tree with `N` levels because
 *  the internal representation is a bit pattern, which sets the bit if the
 *  swap is done. `N` is limited by the amount of Bits in `BitPatternT`.
 *  If `operator[]` is called once, no function to change the `do_swap`
 *  array should be called by the user. The state of the class is then a
 *  read-only mode. Moreover the array `i` can only be read successively
 *  by the `operator[]`. Thus reading the pattern starts with a specific
 *  `j` and in the next call `j-1` must be used as a function argument.
 *  This is an highly unsafe class design, which is chosen to avoid the
 *  usage of additional registers. E.g. the safety of the class can be
 *  greatly improved if one register is added, which is avoided here
 *  due to performance reasons.
 *
 *  \note When `std::bitset` is supported for device code. This should be
 *        used to save the bit pattern
 *  \note When compiled for device code CUDA currently supports built-in
 *        function to count leading zeros `clz` for `int` and `long long int`, which
 *        then must be used for `BitPatternT`.
 */
template <class BitPatternT>
class bit_successive_permutation_t {

   public:
	CUDA_FHD bit_successive_permutation_t(int N) : bit_pattern_(BitPatternT(1) << (N-1)) {
#ifndef NDEBUG
		N_ = N;
#endif
	}

	CUDA_FHD void do_swap(int j) {
		gsl_Expects(j < sizeof(BitPatternT) * 8);
#ifndef NDEBUG
		gsl_Expects(!read_only_mode_);
		gsl_Expects(j < N_ - 1);
#endif
		bit_pattern_ &= ~(BitPatternT(1) << j);
	}

	CUDA_FHD void do_not_swap(int j) {
		gsl_Expects(j < sizeof(BitPatternT) * 8);
#ifndef NDEBUG
		gsl_Expects(!read_only_mode_);
		gsl_Expects(j < N_ - 1);
#endif
		bit_pattern_ |= (BitPatternT(1) << j);
	}

	CUDA_FHD void set(int j, bool do_swap_) {
		if (do_swap_) {
			do_swap(j);
		}
		else {
			do_not_swap(j);
		}
	}

	CUDA_FHD int operator[](int i) {
		gsl_Expects(i < sizeof(BitPatternT) * 8);
#ifndef NDEBUG
		gsl_Expects(i < N_);
		if (!read_only_mode_) {
			read_only_mode_ = true;
			last_i_ = N_;
		}
		gsl_Expects(i == last_i_ - 1);
		last_i_ = i;
#endif
		gsl_Expects(i >= 0);
		const bool flag = (1ll << i) & bit_pattern_;
		bit_pattern_ &= ~(1ll << i);

		if (flag) {
			const int lz = count_leading_zeros(bit_pattern_);
			return sizeof(BitPatternT) * 8 - lz;
		}
		else {
			return i + 1;
		}
	}

   private:
#ifndef NDEBUG
	bool read_only_mode_ = false;
	int N_;
	int last_i_;
#endif
	BitPatternT bit_pattern_;
};

} // namespace permutation

} // namespace thrustshift
