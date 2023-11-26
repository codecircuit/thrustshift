#pragma once

#include <thrustshift/defines.h>

namespace thrustshift {

//! Allocate CUDA managed memory
template <typename T>
class managed_allocator {
   public:
	using value_type = T;

	constexpr managed_allocator(void) noexcept {
	}
	template <typename U>
	constexpr managed_allocator(managed_allocator<U> const&) noexcept {
	}

	value_type* allocate(std::size_t n) {

		const size_t bytes = n * sizeof(T);
		void* p = nullptr;
		THRUSTSHIFT_CHECK_CUDA_ERROR(cudaMallocManaged(&p, bytes));
		if (p == nullptr)
			throw std::bad_alloc{};
		return reinterpret_cast<value_type*>(p);
	}

	void deallocate(value_type* p, std::size_t) noexcept {
		THRUSTSHIFT_CHECK_CUDA_ERROR(cudaFree(p));
	}
};

template <typename T, typename U>
bool operator==(managed_allocator<T> const&,
                managed_allocator<U> const&) noexcept {
	return true;
}
template <typename T, typename U>
bool operator!=(managed_allocator<T> const& x,
                managed_allocator<U> const& y) noexcept {
	return !(x == y);
}

} // namespace thrustshift
