#pragma once

#include <iterator>
#include <type_traits>
#include <utility>

#include <cuda/std/array>

#include <thrust/tuple.h>

#include <thrustshift/container-conversion.h>
#include <thrustshift/defines.h>
#include <thrustshift/transform.h>
#include <thrustshift/type-traits.h>

namespace thrustshift {

template <class Ref, std::size_t N>
using multi_iterator_reference = typename make_thrust_tuple_type<Ref, N>::type;

namespace detail {

template <typename It,
          std::size_t N,
          template <typename, std::size_t>
          class Arr,
          std::size_t... I>
THRUSTSHIFT_FHD auto array_of_iterators2reference_thrust_tuple_impl(
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
THRUSTSHIFT_FHD auto array_of_iterators2reference_thrust_tuple(
    const Arr<It, N>& arr) {
	return detail::array_of_iterators2reference_thrust_tuple_impl(arr, I{});
}

template <class It,
          std::size_t N,
          class Ref = multi_iterator_reference<
              typename std::iterator_traits<It>::reference,
              N>>
class multi_iterator {

   public:
	using reference = Ref;
	using difference_type = typename std::iterator_traits<It>::difference_type;

	using iterator_category =
	    typename std::iterator_traits<It>::iterator_category;

	THRUSTSHIFT_FHD
	multi_iterator(const cuda::std::array<It, N>& iterators)
	    : iterators_(iterators) {
	}
	THRUSTSHIFT_FHD
	multi_iterator(cuda::std::array<It, N>&& iterators)
	    : iterators_(std::move(iterators)) {
	}

	THRUSTSHIFT_FHD constexpr std::size_t size() const noexcept {
		return iterators_.size();
	}

	THRUSTSHIFT_FHD reference operator*() {
		return array_of_iterators2reference_thrust_tuple(iterators_);
	}

	THRUSTSHIFT_FHD multi_iterator<It, N, Ref>& operator+=(
	    difference_type index) {
		iterators_ = array::transform(
		    iterators_,
		    [index] THRUSTSHIFT_HD(const It& it) { return it + index; });
		return *this;
	}

	THRUSTSHIFT_FHD multi_iterator<It, N, Ref>& operator-=(
	    difference_type index) {
		iterators_ = array::transform(
		    iterators_,
		    [index] THRUSTSHIFT_HD(const It& it) { return it - index; });
		return *this;
	}

	THRUSTSHIFT_FHD multi_iterator<It, N, Ref>& operator++() {
		iterators_ = array::transform(
		    iterators_, [] THRUSTSHIFT_HD(const It& it) { return it + 1; });
		return *this;
	}

	THRUSTSHIFT_FHD reference operator[](difference_type index) const {
		auto it = *this;
		it += index;
		return *it;
	}

   private:
	cuda::std::array<It, N> iterators_;
};

//
// Define free operators +,-
//

////// + //////
template <
    class It,
    std::size_t N,
    class difference_type = typename multi_iterator<It, N>::difference_type>
THRUSTSHIFT_FHD multi_iterator<It, N> operator+(const multi_iterator<It, N>& it,
                                                difference_type n) {
	multi_iterator<It, N> other(it);
	other += n;
	return other;
}

template <
    class It,
    std::size_t N,
    class difference_type = typename multi_iterator<It, N>::difference_type>
THRUSTSHIFT_FHD multi_iterator<It, N> operator+(
    difference_type n,
    const multi_iterator<It, N>& it) {
	return it + n;
}

////// - //////
template <
    class It,
    std::size_t N,
    class difference_type = typename multi_iterator<It, N>::difference_type>
THRUSTSHIFT_FHD multi_iterator<It, N> operator-(const multi_iterator<It, N>& it,
                                                difference_type n) {
	multi_iterator<It, N> other(it);
	other -= n;
	return other;
}

} // namespace thrustshift
