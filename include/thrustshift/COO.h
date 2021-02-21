#pragma once

#include <memory_resource>
#include <vector>

#include <thrust/execution_policy.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sort.h>
#include <thrust/tuple.h>

#include <gsl-lite/gsl-lite.hpp>

#include <cuda/define_specifiers.hpp>

#include <thrustshift/managed-vector.h>
#include <thrustshift/memory-resource.h>

namespace thrustshift {

enum class storage_order_t { row_major, col_major, none };

template <typename DataType, typename IndexType>
class COO {
   private:
	pmr::managed_resource_type default_resource_;

   public:
	using value_type = DataType;
	using index_type = IndexType;

	COO() : num_cols_(0), num_rows_(0) {
	}

	template <class DataRange,
	          class RowIndRange,
	          class ColIndRange,
	          class MemoryResource>
	COO(DataRange&& values,
	    RowIndRange&& row_indices,
	    ColIndRange&& col_indices,
	    size_t num_rows,
	    size_t num_cols,
	    MemoryResource& memory_resource)
	    : values_(values.begin(), values.end(), &memory_resource),
	      row_indices_(row_indices.begin(),
	                   row_indices.end(),
	                   &memory_resource),
	      col_indices_(col_indices.begin(),
	                   col_indices.end(),
	                   &memory_resource),
	      num_rows_(num_rows),
	      num_cols_(num_cols),
	      storage_order_(storage_order_t::none) {
		gsl_Expects(values.size() == row_indices.size());
		gsl_Expects(values.size() == col_indices.size());
	}

	template <class DataRange, class ColIndRange, class RowIndRange>
	COO(DataRange&& values,
	    RowIndRange&& row_indices,
	    ColIndRange&& col_indices,
	    size_t num_rows,
	    size_t num_cols)
	    : COO(std::forward<DataRange>(values),
	          std::forward<RowIndRange>(row_indices),
	          std::forward<ColIndRange>(col_indices),
	          num_rows,
	          num_cols,
	          default_resource_) {
	}

	template <class CSR, class MemoryResource>
	COO(CSR&& csr_mtx, MemoryResource& memory_resource)
	    : values_(csr_mtx.values().begin(),
	              csr_mtx.values().end(),
	              &memory_resource),
	      row_indices_(csr_mtx.col_indices().size(), &memory_resource),
	      col_indices_(csr_mtx.col_indices().begin(),
	                   csr_mtx.col_indices().end(),
	                   &memory_resource),
	      num_rows_(csr_mtx.num_rows()),
	      num_cols_(csr_mtx.num_cols()),
	      storage_order_(storage_order_t::none) {

		for (size_t row_id = 0; row_id < csr_mtx.row_ptrs().size() - 1;
		     ++row_id) {
			// nns = not null space
			for (size_t nns_id = csr_mtx.row_ptrs()[row_id];
			     nns_id < csr_mtx.row_ptrs()[row_id + 1];
			     ++nns_id) {
				row_indices_[nns_id] = row_id;
			}
		}
	}

	COO(const COO& other) = default;

	COO(COO&& other) = default;

	void change_storage_order(storage_order_t new_storage_order) {
		IndexType* keys_first;
		IndexType* keys_last;
		IndexType* other_indices_start;

		switch (new_storage_order) {
			case storage_order_t::row_major:
				keys_first = row_indices_.data();
				keys_last = row_indices_.data() + row_indices_.size();
				other_indices_start = col_indices_.data();
				break;
			case storage_order_t::col_major:
				keys_first = col_indices_.data();
				keys_last = col_indices_.data() + col_indices_.size();
				other_indices_start = row_indices_.data();
				break;
			case storage_order_t::none:
				return;
		}

		auto value_it = thrust::make_zip_iterator(
		    thrust::make_tuple(other_indices_start, values_.data()));
		thrust::sort_by_key(thrust::cuda::par, keys_first, keys_last, value_it);
		storage_order_ = new_storage_order;
	}

	void transpose() {
		std::swap(col_indices_, row_indices_);
		std::swap(num_rows_, num_cols_);
		switch (storage_order_) {
			case storage_order_t::none:
				break;
			case storage_order_t::row_major:
				storage_order_ = storage_order_t::col_major;
				break;
			case storage_order_t::col_major:
				storage_order_ = storage_order_t::row_major;
				break;
		}
	}

	gsl_lite::span<DataType> values() {
		return gsl_lite::make_span(values_);
	}

	gsl_lite::span<const DataType> values() const {
		return gsl_lite::make_span(values_);
	}

	gsl_lite::span<IndexType> row_indices() {
		return gsl_lite::make_span(row_indices_);
	}

	gsl_lite::span<const IndexType> row_indices() const {
		return gsl_lite::make_span(row_indices_);
	}

	gsl_lite::span<IndexType> col_indices() {
		return gsl_lite::make_span(col_indices_);
	}

	gsl_lite::span<const IndexType> col_indices() const {
		return gsl_lite::make_span(col_indices_);
	}

	size_t num_rows() const {
		return num_rows_;
	}

	size_t num_cols() const {
		return num_cols_;
	}

	storage_order_t get_storage_order() const {
		return storage_order_;
	}

	//! Return row_ptrs or col_ptrs depending on the current storage order
	thrustshift::managed_vector<index_type> get_ptrs() const {
		gsl_Expects(storage_order_ != storage_order_t::none);
		auto indices = storage_order_ == storage_order_t::row_major
		                   ? row_indices()
		                   : col_indices();
		auto size = storage_order_ == storage_order_t::row_major ? num_rows()
		                                                         : num_cols();

		if (indices.size() > 0) {
			thrustshift::managed_vector<index_type> ptrs(size + 1, -1);
			for (int nns_id = gsl_lite::narrow<int>(indices.size()) - 1; nns_id >= 0; --nns_id) {
				ptrs[indices[nns_id]] = nns_id;
			}
			ptrs[0] = 0;
			ptrs.back() = indices.size();
			index_type k = indices.size();
			for (int id = gsl_lite::narrow<int>(ptrs.size()) - 1; id > 0; --id) {
				if (ptrs[id] == -1) {
					ptrs[id] = k;
				}
				else {
					k = ptrs[id];
				}
			}
			return ptrs;
		}
		else {
			return {0};
		}
	}

   private:
	std::pmr::vector<DataType> values_;
	std::pmr::vector<IndexType> row_indices_;
	std::pmr::vector<IndexType> col_indices_;
	size_t num_rows_;
	size_t num_cols_;
	storage_order_t storage_order_;
};

template <typename DataType, typename IndexType>
class COO_view {

   public:
	template <typename OtherDataType, typename OtherIndexType>
	COO_view(COO<OtherDataType, OtherIndexType>& owner)
	    : values_(owner.values()),
	      row_indices_(owner.row_indices()),
	      col_indices_(owner.col_indices()),
	      num_rows_(owner.num_rows()),
	      num_cols_(owner.num_cols()) {
	}

	template <typename OtherDataType, typename OtherIndexType>
	COO_view(const COO<OtherDataType, OtherIndexType>& owner)
	    : values_(owner.values()),
	      row_indices_(owner.row_indices()),
	      col_indices_(owner.col_indices()),
	      num_rows_(owner.num_rows()),
	      num_cols_(owner.num_cols()) {
	}

	COO_view(const COO_view& other) = default;

	CUDA_FHD gsl_lite::span<DataType> values() {
		return gsl_lite::make_span(values_);
	}

	CUDA_FHD gsl_lite::span<const DataType> values() const {
		return gsl_lite::make_span(values_);
	}

	CUDA_FHD gsl_lite::span<IndexType> row_indices() {
		return gsl_lite::make_span(row_indices_);
	}

	CUDA_FHD gsl_lite::span<const IndexType> row_indices() const {
		return gsl_lite::make_span(row_indices_);
	}

	CUDA_FHD gsl_lite::span<IndexType> col_indices() {
		return gsl_lite::make_span(col_indices_);
	}

	CUDA_FHD gsl_lite::span<const IndexType> col_indices() const {
		return gsl_lite::make_span(col_indices_);
	}

	CUDA_FHD size_t num_rows() const {
		return num_rows_;
	}

	CUDA_FHD size_t num_cols() const {
		return num_cols_;
	}

   private:
	gsl_lite::span<DataType> values_;
	gsl_lite::span<IndexType> row_indices_;
	gsl_lite::span<IndexType> col_indices_;
	size_t num_rows_;
	size_t num_cols_;
};

} // namespace thrustshift
