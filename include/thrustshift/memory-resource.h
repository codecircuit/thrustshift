#pragma once

#include <map>
#include <vector>
#include <memory_resource>

#include <gsl-lite/gsl-lite.hpp>

#include <cuda/runtime_api.hpp>

namespace thrustshift {

namespace pmr {

//! Unified CUDA Memory Resource
class managed_resource_type : public std::pmr::memory_resource {
	void* do_allocate(std::size_t bytes, std::size_t alignment) override {
		auto region = cuda::memory::managed::detail::allocate(bytes);
		return region.get();
	}

	void do_deallocate(void* p,
	                   std::size_t bytes,
	                   std::size_t alignment) override {
		cuda::memory::managed::detail::free(p);
	}

	bool do_is_equal(const std::pmr::memory_resource& other) const noexcept override {
		return this == &other;
	}
};

//! Device CUDA Memory Resource
class device_resource_type : public std::pmr::memory_resource {
	void* do_allocate(std::size_t bytes, std::size_t alignment) override {
		auto region = cuda::memory::device::detail::allocate(bytes);
		return region.get();
	}

	void do_deallocate(void* p,
	                   std::size_t bytes,
	                   std::size_t alignment) override {
		cuda::memory::device::free(p);
	}

	bool do_is_equal(const std::pmr::memory_resource& other) const noexcept override {
		return this == &other;
	}
};


/*! \brief Allocates only if requested buffer size was not allocated previously.
 *
 *  This pool is useful for functions, which require temporary GPU memory. The
 *  host can allocate memory via this pool, launch the kernel with the corresponding
 *  pointer and exit the function without deallocating the memory because the
 *  host is not aware about the runtime of the GPU kernel and when the kernel
 *  may read the temporary required memory. This pool should not be used if the
 *  byte size of the allocations differ often becaues every time an allocation
 *  is called it actually results into a new allocation if not the exact same
 *  size was allocated previously.
 *
 *  Do not use if different buffers of the same size are required.
 *
 *  \note std::pmr compatible
 */
template <class Upstream>
class oversubscribed_delayed_pool_type : public std::pmr::memory_resource {
   private:
	struct book_page_type {
		void* ptr;
		size_t alignment;
	};

   public:
	oversubscribed_delayed_pool_type() = default;

	~oversubscribed_delayed_pool_type() noexcept {
		for (auto& [bytes, book_page] : book_) {
			res_.deallocate(book_page.ptr, bytes, book_page.alignment);
		}
	}

   private:
	void* do_allocate(size_t bytes, size_t alignment) override {
		if (auto it = book_.find(bytes); it != book_.end()) {
			return it->second.ptr;
		}
		void* ptr = res_.allocate(bytes, alignment);
		book_[bytes] = {ptr, alignment};
		return ptr;
	}

	void do_deallocate([[maybe_unused]] void* ptr,
	                   [[maybe_unused]] size_t bytes,
	                   [[maybe_unused]] size_t alignment) noexcept override {
	}

	bool do_is_equal(
	    const std::pmr::memory_resource& other) const noexcept override {
		return this == &other;
	}

	Upstream res_;
	// size in bytes -> (ptr, alignment)
	std::map<size_t, book_page_type> book_;
};

/*! \brief Allocates only if requested buffer size is currently not in the free pool.
 *
 *  This pool is useful for functions, which require temporary GPU memory. The
 *  host can allocate memory via this pool, launch the kernel with the corresponding
 *  pointer and exit the function without deallocating the memory because the
 *  host is not aware about the runtime of the GPU kernel and when the kernel
 *  may read the temporary required memory. This pool should not be used if the
 *  byte size of the allocations differ often becaues every time an allocation
 *  is called it actually results into a new allocation if not the exact same
 *  size was allocated previously.
 *  
 *  Contrary to `oversubscribed_delayed_pool_type` this resource does not return
 *  the same buffer if `deallocate` was not called for that buffer previously. This
 *  is useful if functions delegate the memory resource to other functions, which
 *  are maybe called iteratively and if the function requires different buffers of
 *  the same size.
 *
 *  \note std::pmr compatible
 */
template <class Upstream>
class delayed_pool_type : public std::pmr::memory_resource {
   private:

	struct page_item_type {
		void* ptr;
		size_t alignment;
		bool allocated;
	};

   public:
	delayed_pool_type() = default;

	~delayed_pool_type() noexcept {
		for (auto& [bytes, book_page] : book_) {
			for (auto& page_item : book_page) {
				res_.deallocate(page_item.ptr, bytes, page_item.alignment);
			}
		}
	}

   private:
	void* do_allocate(size_t bytes, size_t alignment) override {
		if (bytes == 0) {
			return nullptr;
		}
		if (auto it = book_.find(bytes); it != book_.end()) {
			for (auto& page_item : it->second) {
				if (!page_item.allocated) {
					page_item.allocated = true;
					return page_item.ptr;
				}
			}
			// no page item was free
			void* ptr = res_.allocate(bytes, alignment);
			it->second.push_back({ptr, alignment, true});
			return ptr;
		}
		// the required size was never allocated before
		void* ptr = res_.allocate(bytes, alignment);
		book_[bytes] = {{ptr, alignment, true}};
		return ptr;
	}

	void do_deallocate(void* ptr,
	                   size_t bytes,
	                   size_t alignment) noexcept override {
		if (ptr == nullptr) {
			return;
		}
		for (auto& page_item : book_[bytes]) {
			if (page_item.ptr == ptr && page_item.allocated) {
				gsl_Expects(page_item.alignment == alignment);
				page_item.allocated = false;
				return;
			}
		}
		// The pointer which should be deallocated was never allocated before
		std::terminate();
	}

	bool do_is_equal(
	    const std::pmr::memory_resource& other) const noexcept override {
		return this == &other;
	}

	Upstream res_;
	// size in bytes -> (ptr, alignment)
	std::map<size_t, std::vector<page_item_type>> book_;
};

} // namespace pmr

} // namespace thrustshift
