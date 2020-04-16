#pragma once

#include <type_traits>

#include <thrust/execution_policy.h>
#include <thrust/gather.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/scatter.h>

#include <gsl-lite/gsl-lite.hpp>

#include <thrustshift/managed-vector.h>

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

} // namespace permutation

} // namespace thrustshift
