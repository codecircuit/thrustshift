#pragma once

#include <cuda/runtime_api.hpp>

namespace thrustshift {

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
		auto mem = cuda::memory::managed::detail::allocate(n * sizeof(T)).get();
		if (mem == nullptr)
			throw std::bad_alloc{};
		return static_cast<value_type*>(mem);
	}

	void deallocate(value_type* p, std::size_t) noexcept {
		cuda::memory::managed::detail::free(p);
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

}
