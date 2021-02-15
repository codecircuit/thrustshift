#pragma once

#include <memory>
#include <memory_resource>

#include <gsl-lite/gsl-lite.hpp>

#include <sysmakeshift/memory.hpp>

namespace thrustshift {

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
	    : ptr_(sysmakeshift::allocate_unique<T[]>(alloc, size)), size_(size) {
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

} // namespace thrustshift
