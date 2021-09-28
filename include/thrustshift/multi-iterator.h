#pragma once

#include <iterator>
#include <type_traits>
#include <utility>

#include <cuda/define_specifiers.hpp>

#include <thrust/tuple.h>

#include <kat/containers/array.hpp>

#include <thrustshift/transform.h>
#include <thrustshift/type-traits.h>
#include <thrustshift/container-conversion.h>

namespace thrustshift {

// 1. Use a tuple to represent the references, array does not work, as an array must represent
//    continous lying data.
// 2. Use a vector type, which can be constructed from tuples or tuple references or arrays, to
//    implement the arithmetic operations. Later assign the tuple references to the vector with the
//    arithmetic result.
//

template <class Ref, std::size_t N>
using multi_iterator_reference = typename make_thrust_tuple_type<Ref, N>::type;

namespace detail {

template <typename It,
          std::size_t N,
          template <typename, std::size_t>
          class Arr,
          std::size_t... I>
CUDA_FHD auto array_of_iterators2reference_thrust_tuple_impl(
    const Arr<It, N>& arr,
    std::index_sequence<I...>) {
	using Ref = typename std::iterator_traits<It>::reference;
	using TupleT = typename make_thrust_tuple_type<Ref, N>::type;
	return TupleT((*arr[I])...);
}

} // namespace detail

template <typename It,
          std::size_t N,
          template <typename, std::size_t>
          class Arr,
          class I = std::make_index_sequence<N>>
CUDA_FHD auto array_of_iterators2reference_thrust_tuple(const Arr<It, N>& arr) {
	return detail::array_of_iterators2reference_thrust_tuple_impl(arr, I{});
}

template <class It, std::size_t N, class Ref = multi_iterator_reference<typename std::iterator_traits<It>::reference, N>>
class multi_iterator {

   public:
	using reference = Ref;
	using difference_type = typename std::iterator_traits<It>::difference_type;

	using iterator_category =
	    typename std::iterator_traits<It>::iterator_category;

	CUDA_FHD
	multi_iterator(const kat::array<It, N>& iterators) : iterators_(iterators) {
	}
	CUDA_FHD
	multi_iterator(kat::array<It, N>&& iterators)
	    : iterators_(std::move(iterators)) {
	}

	CUDA_FHD constexpr std::size_t size() const noexcept {
		return iterators_.size();
	}

	CUDA_FHD reference operator*() {
		return array_of_iterators2reference_thrust_tuple(iterators_);
	}

	CUDA_FHD multi_iterator<It, N, Ref>& operator+=(difference_type index) {
		iterators_ = array::transform(
		    iterators_, [index] CUDA_HD(const It& it) { return it + index; });
		return *this;
	}

	CUDA_FHD multi_iterator<It, N, Ref>& operator-=(difference_type index) {
		iterators_ = array::transform(
		    iterators_, [index] CUDA_HD(const It& it) { return it - index; });
		return *this;
	}

	CUDA_FHD multi_iterator<It, N, Ref>& operator++() {
		iterators_ = array::transform(
		    iterators_, [] CUDA_HD(const It& it) { return it + 1; });
		return *this;
	}

	CUDA_FHD reference operator[](difference_type index) const {
		auto it = *this;
		it += index;
		return *it;
	}

   private:
	kat::array<It, N> iterators_;
};


//
// Define free operators +,-
//

////// + //////
template <
    class It,
    std::size_t N,
    class difference_type = typename multi_iterator<It, N>::difference_type>
CUDA_FHD multi_iterator<It, N> operator+(const multi_iterator<It, N>& it,
                                         difference_type n) {
	multi_iterator<It, N> other(it);
	other += n;
	return other;
}

template <
    class It,
    std::size_t N,
    class difference_type = typename multi_iterator<It, N>::difference_type>
CUDA_FHD multi_iterator<It, N> operator+(difference_type n,
                                         const multi_iterator<It, N>& it) {
	return it + n;
}

////// - //////
template <
    class It,
    std::size_t N,
    class difference_type = typename multi_iterator<It, N>::difference_type>
CUDA_FHD multi_iterator<It, N> operator-(const multi_iterator<It, N>& it,
                                         difference_type n) {
	multi_iterator<It, N> other(it);
	other -= n;
	return other;
}

} // namespace thrustshift
