#pragma once

#include <cuda/define_specifiers.hpp>

/*! \file functional.h
 *  \brief Useful when device lambdas result into compliler errors
 */

namespace thrustshift {

template <typename Reference, typename B>
struct assign_equal {

	CUDA_FHD Reference operator()(Reference a, const B& b) {
		a = b;
		return a;
	}
};

template <typename Reference, typename B>
struct plus_equal_assign {

	CUDA_FHD Reference operator()(Reference a, const B& b) {
		a += b;
		return a;
	}
};

template <typename Reference, typename B>
struct minus_equal_assign {

	CUDA_FHD Reference operator()(Reference a, const B& b) {
		a -= b;
		return a;
	}
};

template <typename Reference, typename B>
struct multiply_equal_assign {

	CUDA_FHD Reference operator()(Reference a, const B& b) {
		a *= b;
		return a;
	}
};

template <typename Reference, typename B>
struct divide_equal_assign {

	CUDA_FHD Reference operator()(Reference a, const B& b) {
		a /= b;
		return a;
	}
};

template <typename Reference, typename B>
struct plus_equal_assign_constant {

	CUDA_FHD plus_equal_assign_constant(const B& b) : b_(b) {
	}

	CUDA_FHD Reference operator()(Reference a) {
		a += b_;
		return a;
	}

	B b_;
};

template <typename Reference, typename B>
struct left_plus_equal_assign_constant {

	CUDA_FHD left_plus_equal_assign_constant(const B& b) : b_(b) {
	}

	CUDA_FHD Reference operator()(Reference a) {
		a = b_ + a;
		return a;
	}
	B b_;
};

template <typename Reference, typename B>
struct left_multiply_equal_assign_constant {

	CUDA_FHD left_multiply_equal_assign_constant(const B& b) : b_(b) {
	}

	CUDA_FHD Reference operator()(Reference a) {
		a = b_ * a;
		return a;
	}
	B b_;
};

} // namespace thrustshift
