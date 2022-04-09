#pragma once

#include <cstddef> // std::byte
#include <map>
#include <memory> // std::align
#include <memory_resource>
#include <vector>

#include <gsl-lite/gsl-lite.hpp>

#include <cuda/runtime_api.hpp>

namespace thrustshift {

namespace pmr {

//! Unified CUDA Memory Resource. Buffers cannet be deallocated partially. Alignment is ignored.
class managed_resource_type : public std::pmr::memory_resource {
	void* do_allocate(std::size_t bytes,
	                  [[maybe_unused]] std::size_t alignment) override {
		auto device = cuda::device::current::get();
		auto region = cuda::memory::managed::allocate(device, bytes);
		return region.get();
	}

	void do_deallocate(void* p,
	                   [[maybe_unused]] std::size_t bytes,
	                   [[maybe_unused]] std::size_t alignment) override {
		cuda::memory::managed::free(p);
	}

	bool do_is_equal(
	    const std::pmr::memory_resource& other) const noexcept override {
		return this == &other;
	}
};

//! Device CUDA Memory Resource. Buffers cannot be deallocated partially. Alignment is ignored.
class device_resource_type : public std::pmr::memory_resource {
	void* do_allocate(std::size_t bytes,
	                  [[maybe_unused]] std::size_t alignment) override {
		auto device = cuda::device::current::get();
		auto region = cuda::memory::device::allocate(device, bytes);
		return region.get();
	}

	void do_deallocate(void* p,
	                   [[maybe_unused]] std::size_t bytes,
	                   [[maybe_unused]] std::size_t alignment) override {
		cuda::memory::device::free(p);
	}

	bool do_is_equal(
	    const std::pmr::memory_resource& other) const noexcept override {
		return this == &other;
	}
};

class host_resource_type : public std::pmr::memory_resource {
	void* do_allocate(std::size_t bytes,
	                  [[maybe_unused]] std::size_t alignment) override {
		return malloc(bytes);
	}

	void do_deallocate(void* p,
	                   [[maybe_unused]] std::size_t bytes,
	                   [[maybe_unused]] std::size_t alignment) override {
		free(p);
	}

	bool do_is_equal(
	    const std::pmr::memory_resource& other) const noexcept override {
		return this == &other;
	}
};

class pinned_host_resource_type : public std::pmr::memory_resource {
	void* do_allocate(std::size_t bytes,
	                  [[maybe_unused]] std::size_t alignment) override {
		void* ptr = nullptr;
		cuda::throw_if_error(cudaMallocHost(&ptr, bytes),
		                     "failed allocating pinned host memory");
		return ptr;
	}

	void do_deallocate(void* p,
	                   [[maybe_unused]] std::size_t bytes,
	                   [[maybe_unused]] std::size_t alignment) override {
		cuda::throw_if_error(cudaFreeHost(p), "failed cudaFreeHost");
	}

	bool do_is_equal(
	    const std::pmr::memory_resource& other) const noexcept override {
		return this == &other;
	}
};

namespace detail {

struct page_id_type {
	size_t bytes;
	size_t alignment;
};

inline bool operator<(const page_id_type& a, const page_id_type& b) {
	if (a.bytes == b.bytes) {
		return a.alignment < b.alignment;
	}
	return a.bytes < b.bytes;
}

} // namespace detail

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
 *  This class is useful if functions delegate the memory resource to other functions, which
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
		bool allocated;
	};

   public:
	delayed_pool_type() = default;

	delayed_pool_type(delayed_pool_type&&) = default;

	~delayed_pool_type() noexcept {
		for (auto& [page_id, book_page] : book_) {
			for (auto& page_item : book_page) {
				res_.deallocate(
				    page_item.ptr, page_id.bytes, page_id.alignment);
			}
		}
	}

	auto& get_book() const {
		return book_;
	}

   private:
	void* do_allocate(size_t bytes, size_t alignment) override {
		if (bytes == 0) {
			return nullptr;
		}
		const detail::page_id_type page_id({bytes, alignment});
		if (auto it = book_.find(page_id); it != book_.end()) {
			for (auto& page_item : it->second) {
				if (!page_item.allocated) {
					page_item.allocated = true;
					return page_item.ptr;
				}
			}
			// no page item was free
			void* ptr = res_.allocate(bytes, alignment);
			it->second.push_back({ptr, true});
			return ptr;
		}
		// the required size was never allocated before
		void* ptr = res_.allocate(bytes, alignment);
		book_[page_id] = {{ptr, true}};
		return ptr;
	}

	void do_deallocate(void* ptr,
	                   size_t bytes,
	                   size_t alignment) noexcept override {
		const detail::page_id_type page_id({bytes, alignment});
		if (ptr == nullptr || bytes == 0) {
			return;
		}
		for (auto& page_item : book_[page_id]) {
			if (page_item.ptr == ptr && page_item.allocated) {
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
	std::map<detail::page_id_type, std::vector<page_item_type>> book_;
};

/*! \brief Allocates only if there is no buffer in the free pool which is of the required size or larger.
 *
 *  This pool searches for the smallest buffer in the pool which has the required alignment and is
 *  of equal or larger byte size than the required buffer and returns that buffer if do_allocate is called.
 *  Morevover, this pool has the same 'delayed' properties than the `delayed_pool_type`.
 *
 *  \sa delayed_pool_type
 *  \note std::pmr compatible
 */
template <class Upstream>
class delayed_fragmenting_pool_type : public std::pmr::memory_resource {
   private:
	struct page_item_type {
		void* ptr;
		bool allocated;
	};

   public:
	delayed_fragmenting_pool_type() = default;

	delayed_fragmenting_pool_type(delayed_fragmenting_pool_type&&) = default;

	~delayed_fragmenting_pool_type() noexcept {
		for (auto& [page_id, book_page] : book_) {
			for (auto& page_item : book_page) {
				res_.deallocate(
				    page_item.ptr, page_id.bytes, page_id.alignment);
			}
		}
	}

	auto& get_book() const {
		return book_;
	}

   private:
	void* do_allocate(size_t bytes, size_t alignment) override {
		if (bytes == 0) {
			return nullptr;
		}
		const detail::page_id_type page_id({bytes, alignment});
		for (auto& [k, v] : book_) {
			if (k.alignment == alignment && k.bytes >= bytes) {
				for (auto& p : v) {
					if (!p.allocated) {
						p.allocated = true;
						return p.ptr;
					}
				}
			}
		}
		// no page item was free
		if (auto it = book_.find(page_id); it != book_.end()) {
			void* ptr = res_.allocate(bytes, alignment);
			it->second.push_back({ptr, true});
			return ptr;
		}
		// the required size was never allocated before
		void* ptr = res_.allocate(bytes, alignment);
		book_[page_id] = {{ptr, true}};
		return ptr;
	}

	void do_deallocate(void* ptr,
	                   size_t bytes,
	                   size_t alignment) noexcept override {
		const detail::page_id_type page_id({bytes, alignment});

		if (ptr == nullptr || bytes == 0) {
			return;
		}

		for (auto& [k, v] : book_) {
			if (k.alignment == alignment && k.bytes >= bytes) {
				for (auto& p : v) {
					// no comparison based on the bytes because the buffer can be larger
					// than the amount of requested bytes.
					if (p.ptr == ptr && p.allocated) {
						p.allocated = false;
						return;
					}
				}
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
	std::map<detail::page_id_type, std::vector<page_item_type>> book_;
};

/*! \brief Wraps a memory resource around an existing buffer of fixed size.
 *
 *  This class takes a pointer and a size of an existing buffer and wraps a memory
 *  resource around that buffer. The resource can provide parts of that buffer by
 *  calling `do_allocate` and simply returns subsequently aligned parts of the buffer.
 *  If the buffer size is exceeded `gsl_FailFast()` is called.
 *
 *  \note std::pmr compatible
 */
class wrapping_resource_type : public std::pmr::memory_resource {

   public:
	wrapping_resource_type(void* ptr, size_t size_in_bytes)
	    : size_in_bytes_(size_in_bytes), ptr_(ptr) {
		static_assert(sizeof(std::byte) == 1);
	}

   private:
	void* do_allocate(size_t bytes, size_t alignment) noexcept override {
		if (bytes == 0) {
			return nullptr;
		}
		void* ptr = std::align(alignment, bytes, ptr_, size_in_bytes_);
		gsl_Expects(ptr); // not nullptr
		gsl_Expects(
		    ptr ==
		    ptr_); // this is how I understand the function documentation of std::align
		gsl_Expects(size_in_bytes_ >= bytes);
		size_in_bytes_ -= bytes;
		ptr_ = reinterpret_cast<std::byte*>(ptr_) + bytes;
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

	size_t size_in_bytes_;
	void* ptr_;
};

static managed_resource_type default_resource;

} // namespace pmr

} // namespace thrustshift
