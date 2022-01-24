#pragma once

#include <iterator>
#include <type_traits>
#include <utility>

#include <cuda/define_specifiers.hpp>

#include <thrust/tuple.h>

#include <kat/containers/array.hpp>

#include <thrustshift/container-conversion.h>
#include <thrustshift/for-each.h>
#include <thrustshift/functional.h>
#include <thrustshift/multi-iterator.h>
#include <thrustshift/transform.h>
#include <thrustshift/type-traits.h>

/*! \file arithmetic-multi-iterator-reference.h
 *  \brief Reference type with vector-like arithmetic support
 *
 *  The values are represented in tuples because multiple references of
 *  different iterators cannot be represented as an array of references,
 *  as an array must be saved by consecutive memory. The arithmetic
 *  tuple type also uses tuples to make it easier for the compiler to
 *  convert between different tuple types.
 *
 */
namespace thrustshift {

namespace detail {

template <class Ref, std::size_t N, class Parent>
class arithmetic_multi_iterator_reference_base : public Parent {

   public:
	using parent_t = Parent;
	using parent_t::parent_t;

	using reference_type = Ref;
	using value_type = typename std::decay<reference_type>::type;
	using tuple_value_type =
	    typename make_thrust_tuple_type<value_type, N>::type;

	template <class OtherTuple>
	CUDA_FHD arithmetic_multi_iterator_reference_base<Ref, N, Parent>&
	operator=(OtherTuple&& other) {
		using std::get;
		using other_value_type =
		    typename std::decay<decltype(get<0>(other))>::type;
		tuple::for_each_n<N>(static_cast<parent_t&>(*this),
		                     other,
		                     assign_equal<reference_type, other_value_type>());
		return *this;
	}

	//
	// OPERATORS FOR TUPLES AND SCALARS +=,-=,*=,/=
	// Note: that can be tuples with references **or** normal tuples with values
	//
	template <class T>
	CUDA_FHD arithmetic_multi_iterator_reference_base<Ref, N, Parent>&
	operator+=(T&& t) {
		if constexpr (std::is_convertible<typename std::decay<T>::type,
		                                  value_type>::value) {
			tuple::for_each(static_cast<parent_t&>(*this),
			                [t] CUDA_HD(reference_type x) { x += t; });
		}
		else {
			// for types which are convertible to `tuple_value_type`
			using std::get;
			using other_value_type =
			    typename std::decay<decltype(get<0>(t))>::type;
			tuple::for_each(
			    static_cast<parent_t&>(*this),
			    static_cast<tuple_value_type>(t),
			    plus_equal_assign<reference_type, other_value_type>());
		}
		return *this;
	}

	template <class T>
	CUDA_FHD arithmetic_multi_iterator_reference_base<Ref, N, Parent>&
	operator-=(T&& t) {
		if constexpr (std::is_convertible<typename std::decay<T>::type,
		                                  value_type>::value) {
			tuple::for_each(static_cast<parent_t&>(*this),
			                [t] CUDA_HD(reference_type x) { x -= t; });
		}
		else {
			// for types which are convertible to `tuple_value_type`
			using std::get;
			using other_value_type =
			    typename std::decay<decltype(get<0>(t))>::type;
			tuple::for_each(
			    static_cast<parent_t&>(*this),
			    static_cast<tuple_value_type>(t),
			    minus_equal_assign<reference_type, other_value_type>());
		}
		return *this;
	}

	template <class T>
	CUDA_FHD arithmetic_multi_iterator_reference_base<Ref, N, Parent>&
	operator*=(T&& t) {

		// Alternatively to a `if constexpr` it should be possible to use
		// `std::enable_if`, but I could not get it working correctly.
		if constexpr (std::is_convertible<typename std::decay<T>::type,
		                                  value_type>::value) {
			tuple::for_each(static_cast<parent_t&>(*this),
			                [t] CUDA_HD(reference_type x) { x *= t; });
		}
		else {
			// for types which are convertible to `tuple_value_type`
			using std::get;
			using other_value_type =
			    typename std::decay<decltype(get<0>(t))>::type;
			tuple::for_each(
			    static_cast<parent_t&>(*this),
			    static_cast<tuple_value_type>(t),
			    multiply_equal_assign<reference_type, other_value_type>());
		}
		return *this;
	}

	template <class T>
	CUDA_FHD arithmetic_multi_iterator_reference_base<Ref, N, Parent>&
	operator/=(T&& t) {
		if constexpr (std::is_convertible<typename std::decay<T>::type,
		                                  value_type>::value) {
			tuple::for_each(static_cast<parent_t&>(*this),
			                [t] CUDA_HD(reference_type x) { x /= t; });
		}
		else {
			// for types which are convertible to `tuple_value_type`
			using std::get;
			using other_value_type =
			    typename std::decay<decltype(get<0>(t))>::type;
			tuple::for_each(
			    static_cast<parent_t&>(*this),
			    static_cast<tuple_value_type>(t),
			    divide_equal_assign<reference_type, other_value_type>());
		}
		return *this;
	}
};

} // namespace detail

template <class Ref, std::size_t N>
using arithmetic_multi_iterator_reference =
    detail::arithmetic_multi_iterator_reference_base<
        Ref,
        N,
        multi_iterator_reference<Ref, N>>;

template <class T, std::size_t N>
class arithmetic_tuple
    : public detail::arithmetic_multi_iterator_reference_base<
          T&,
          N,
          typename make_thrust_tuple_type<T, N>::type> {

   public:
	using parent_t = detail::arithmetic_multi_iterator_reference_base<
	    T&,
	    N,
	    typename make_thrust_tuple_type<T, N>::type>;
	using parent_t::parent_t;
};

//
// FREE OPERATORS FOR THE BASE CLASS WITH TUPLES +,-,*,/
//
template <class RefA, class RefB, class ParentA, class ParentB, std::size_t N>
CUDA_FHD auto operator+(
    const detail::arithmetic_multi_iterator_reference_base<RefA, N, ParentA>& a,
    const detail::arithmetic_multi_iterator_reference_base<RefB, N, ParentB>&
        b) {
	using T = typename detail::
	    arithmetic_multi_iterator_reference_base<RefA, N, ParentA>::value_type;
	static_assert(
	    std::is_same<T,
	                 typename detail::arithmetic_multi_iterator_reference_base<
	                     RefB,
	                     N,
	                     ParentB>::value_type>::value);
	arithmetic_tuple<T, N> result{a};
	result += b;
	return result;
}

template <class RefA, class RefB, class ParentA, class ParentB, std::size_t N>
CUDA_FHD auto operator-(
    const detail::arithmetic_multi_iterator_reference_base<RefA, N, ParentA>& a,
    const detail::arithmetic_multi_iterator_reference_base<RefB, N, ParentB>&
        b) {
	using T = typename detail::
	    arithmetic_multi_iterator_reference_base<RefA, N, ParentA>::value_type;
	static_assert(
	    std::is_same<T,
	                 typename detail::arithmetic_multi_iterator_reference_base<
	                     RefB,
	                     N,
	                     ParentB>::value_type>::value);
	arithmetic_tuple<T, N> result{a};
	result -= b;
	return result;
}

template <class RefA, class RefB, class ParentA, class ParentB, std::size_t N>
CUDA_FHD auto operator*(
    const detail::arithmetic_multi_iterator_reference_base<RefA, N, ParentA>& a,
    const detail::arithmetic_multi_iterator_reference_base<RefB, N, ParentB>&
        b) {
	using T = typename detail::
	    arithmetic_multi_iterator_reference_base<RefA, N, ParentA>::value_type;
	static_assert(
	    std::is_same<T,
	                 typename detail::arithmetic_multi_iterator_reference_base<
	                     RefB,
	                     N,
	                     ParentB>::value_type>::value);
	arithmetic_tuple<T, N> result{a};
	result *= b;
	return result;
}

template <class RefA, class RefB, class ParentA, class ParentB, std::size_t N>
CUDA_FHD auto operator/(
    const detail::arithmetic_multi_iterator_reference_base<RefA, N, ParentA>& a,
    const detail::arithmetic_multi_iterator_reference_base<RefB, N, ParentB>&
        b) {
	using T = typename detail::
	    arithmetic_multi_iterator_reference_base<RefA, N, ParentA>::value_type;
	static_assert(
	    std::is_same<T,
	                 typename detail::arithmetic_multi_iterator_reference_base<
	                     RefB,
	                     N,
	                     ParentB>::value_type>::value);
	arithmetic_tuple<T, N> result{a};
	result /= b;
	return result;
}

//
// FREE OPERATORS FOR THE ARITHMETIC TUPLE CLASS +,-,*,/
//
template <typename T, std::size_t N>
CUDA_FHD arithmetic_tuple<T, N> operator+(const arithmetic_tuple<T, N>& a,
                                          const arithmetic_tuple<T, N>& b) {
	arithmetic_tuple<T, N> result{a};
	result += b;
	return result;
}

template <typename T, std::size_t N>
CUDA_FHD arithmetic_tuple<T, N> operator-(const arithmetic_tuple<T, N>& a,
                                          const arithmetic_tuple<T, N>& b) {
	arithmetic_tuple<T, N> result{a};
	result -= b;
	return result;
}

template <typename T, std::size_t N>
CUDA_FHD arithmetic_tuple<T, N> operator*(const arithmetic_tuple<T, N>& a,
                                          const arithmetic_tuple<T, N>& b) {
	arithmetic_tuple<T, N> result{a};
	result *= b;
	return result;
}

template <typename T, std::size_t N>
CUDA_FHD arithmetic_tuple<T, N> operator/(const arithmetic_tuple<T, N>& a,
                                          const arithmetic_tuple<T, N>& b) {
	arithmetic_tuple<T, N> result{a};
	result /= b;
	return result;
}

//
// FREE OPERATORS FOR TUPLE AND VALUE TYPE +,-,*,/
// - tup + scalar
// - scalar + tup
// - tup - scalar
// - tup * scalar
// - scalar * tup
// - tup / scalar
//
template <class Ref,
          std::size_t N,
          class Parent,
          class Scalar,
          std::enable_if_t<
              std::is_convertible<
                  Scalar,
                  typename detail::
                      arithmetic_multi_iterator_reference_base<Ref, N, Parent>::
                          value_type>::value,
              bool> = true>
CUDA_FHD auto operator+(
    const detail::arithmetic_multi_iterator_reference_base<Ref, N, Parent>& tup,
    const Scalar& scalar) {
	using T = typename detail::
	    arithmetic_multi_iterator_reference_base<Ref, N, Parent>::value_type;
	arithmetic_tuple<T, N> result{tup};
	result += scalar;
	return result;
}

template <class Ref,
          std::size_t N,
          class Parent,
          class Scalar,
          std::enable_if_t<
              std::is_convertible<
                  Scalar,
                  typename detail::
                      arithmetic_multi_iterator_reference_base<Ref, N, Parent>::
                          value_type>::value,
              bool> = true>
CUDA_FHD auto operator+(
    const Scalar& scalar,
    const detail::arithmetic_multi_iterator_reference_base<Ref, N, Parent>&
        tup) {
	using T = typename detail::
	    arithmetic_multi_iterator_reference_base<Ref, N, Parent>::value_type;
	arithmetic_tuple<T, N> result{tup};
	using parent_t = typename arithmetic_tuple<T, N>::parent_t::parent_t;
	// A device lambda here in combination with `std::enable_if_t` results into
	// an internal compiler error.
	tuple::for_each(static_cast<parent_t&>(result),
	                left_plus_equal_assign_constant<T&, T>(scalar));
	return result;
}

template <class Ref,
          std::size_t N,
          class Parent,
          class Scalar,
          std::enable_if_t<
              std::is_convertible<
                  Scalar,
                  typename detail::
                      arithmetic_multi_iterator_reference_base<Ref, N, Parent>::
                          value_type>::value,
              bool> = true>
CUDA_FHD auto operator-(
    const detail::arithmetic_multi_iterator_reference_base<Ref, N, Parent>& tup,
    const Scalar& scalar) {
	using T = typename detail::
	    arithmetic_multi_iterator_reference_base<Ref, N, Parent>::value_type;
	arithmetic_tuple<T, N> result{tup};
	result -= scalar;
	return result;
}

template <class Ref,
          std::size_t N,
          class Parent,
          class Scalar,
          std::enable_if_t<
              std::is_convertible<
                  Scalar,
                  typename detail::
                      arithmetic_multi_iterator_reference_base<Ref, N, Parent>::
                          value_type>::value,
              bool> = true>
CUDA_FHD auto operator*(
    const detail::arithmetic_multi_iterator_reference_base<Ref, N, Parent>& tup,
    const Scalar& scalar) {
	using T = typename detail::
	    arithmetic_multi_iterator_reference_base<Ref, N, Parent>::value_type;
	arithmetic_tuple<T, N> result{tup};
	result *= scalar;
	return result;
}

template <class Ref,
          std::size_t N,
          class Parent,
          class Scalar,
          std::enable_if_t<
              std::is_convertible<
                  Scalar,
                  typename detail::
                      arithmetic_multi_iterator_reference_base<Ref, N, Parent>::
                          value_type>::value,
              bool> = true>
CUDA_FHD auto operator*(
    const Scalar& scalar,
    const detail::arithmetic_multi_iterator_reference_base<Ref, N, Parent>&
        tup) {
	using T = typename detail::
	    arithmetic_multi_iterator_reference_base<Ref, N, Parent>::value_type;
	arithmetic_tuple<T, N> result{tup};
	using parent_t = typename arithmetic_tuple<T, N>::parent_t::parent_t;
	// A device lambda here in combination with `std::enable_if_t` results into
	// an internal compiler error.
	tuple::for_each(static_cast<parent_t&>(result),
	                left_multiply_equal_assign_constant<T&, T>(scalar));
	return result;
}

template <class Ref,
          std::size_t N,
          class Parent,
          class Scalar,
          std::enable_if_t<
              std::is_convertible<
                  Scalar,
                  typename detail::
                      arithmetic_multi_iterator_reference_base<Ref, N, Parent>::
                          value_type>::value,
              bool> = true>
CUDA_FHD auto operator/(
    const detail::arithmetic_multi_iterator_reference_base<Ref, N, Parent>& tup,
    const Scalar& scalar) {
	using T = typename detail::
	    arithmetic_multi_iterator_reference_base<Ref, N, Parent>::value_type;
	arithmetic_tuple<T, N> result{tup};
	result /= scalar;
	return result;
}

} // namespace thrustshift
