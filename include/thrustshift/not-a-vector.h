#pragma once

#include <memory>
#include <memory_resource>

#include <gsl-lite/gsl-lite.hpp>

#include <sysmakeshift/memory.hpp>

namespace thrustshift {

namespace detail {

//! Allocate without initialization
template <typename T, typename A, typename SizeC>
T* allocate_array(A& alloc, SizeC sizeC) {
	return std::allocator_traits<A>::allocate(alloc, std::size_t(sizeC));
}

template <typename ArrayT, typename A>
std::enable_if_t<
    sysmakeshift::detail::extent_only<ArrayT>::value == 0,
    std::unique_ptr<ArrayT, sysmakeshift::allocator_deleter<ArrayT, A>>>
allocate_unique(A alloc, std::size_t size) {
	using T = std::remove_cv_t<sysmakeshift::detail::remove_extent_only_t<ArrayT>>;
	static_assert(
	    std::is_same<typename std::allocator_traits<A>::value_type, T>::value,
	    "allocator has mismatching value_type");

	T* ptr = detail::allocate_array<T>(alloc, size);
	return {ptr, {std::move(alloc), size}};
}

} // namespace detail

/*! \brief Container which provides memory with a custom allocator.
 *
 *  Useful in combination with memory which is not accessible on the
 *  host. E.g. you can use a device memory allocator with this class
 *  to obtain device memory and pass it as a span to a GPU kernel.
 */
template <typename T, class Allocator>
class not_a_vector {

   public:
	not_a_vector(std::size_t size, Allocator& alloc)
	    : ptr_(detail::allocate_unique<T[]>(alloc, size)), size_(size) {
	}

	gsl_lite::span<T> to_span() const {
		return gsl_lite::make_span(ptr_.get(), size_);
	}

   private:
	std::unique_ptr<T[], sysmakeshift::allocator_deleter<T[], Allocator>> ptr_;
	std::size_t size_;
};

template <typename T, class Resource>
auto make_not_a_vector(std::size_t size, Resource& memory_resource) {
	std::pmr::polymorphic_allocator<T> alloc(&memory_resource);
	return not_a_vector<T, decltype(alloc)>(size, alloc);
}

/*! \brief Make not_a_vector object and the span on the corresponding buffer.
 *
 *  This function makes it easier to construct the owning container and the view
 *  in one line of code with structured bindings.
 *
 *  ```cpp
 *
 *  auto [nav, span] = make_not_a_vector_and_span<T>(N, resource);
 *  ```
 */
template <typename T, class Resource>
auto make_not_a_vector_and_span(std::size_t size, Resource& memory_resource) {
	std::pmr::polymorphic_allocator<T> alloc(&memory_resource);
	auto nav = not_a_vector<T, decltype(alloc)>(size, alloc);
	auto s = nav.to_span();
	return std::make_tuple(std::move(nav), s);
}

} // namespace thrustshift
