#pragma once

#include <memory_resource>
#include <vector>

#include <gsl-lite/gsl-lite.hpp>

#include <thrustshift/memory-resource.h>

namespace thrustshift {

template <typename DataType, typename IndexType>
class CSR {
   private:
	pmr::managed_resource_type default_resource_;

   public:
	using value_type = DataType;
	using index_type = IndexType;

	CSR() : row_ptrs_(1, 0, &default_resource_), num_cols_(0) {}

	template <class DataRange,
	          class ColIndRange,
	          class RowPtrsRange,
	          class MemoryResource>
	CSR(DataRange&& values,
	    ColIndRange&& col_indices,
	    RowPtrsRange&& row_ptrs,
	    size_t num_cols,
	    MemoryResource& memory_resource)
	    : values_(values.begin(), values.end(), &memory_resource),
	      col_indices_(col_indices.begin(),
	                   col_indices.end(),
	                   &memory_resource),
	      row_ptrs_(row_ptrs.begin(), row_ptrs.end(), &memory_resource),
	      num_cols_(num_cols) {
		gsl_Expects(values.size() == col_indices.size());
		gsl_Expects(row_ptrs.size() > 0);
		gsl_Expects(row_ptrs[0] == 0);
	}

	template <class DataRange, class ColIndRange, class RowPtrsRange>
	CSR(DataRange&& values,
	    ColIndRange&& col_indices,
	    RowPtrsRange&& row_ptrs,
	    size_t num_cols)
	    : CSR(std::forward<DataRange>(values),
	          std::forward<ColIndRange>(col_indices),
	          std::forward<RowPtrsRange>(row_ptrs),
	          num_cols,
	          default_resource_) {
	}

	CSR(const CSR& other) = default;

	CSR(CSR&& other) = default;

	gsl_lite::span<DataType> values() {
		return gsl_lite::make_span(values_);
	}

	gsl_lite::span<const DataType> values() const {
		return gsl_lite::make_span(values_);
	}

	gsl_lite::span<IndexType> col_indices() {
		return gsl_lite::make_span(col_indices_);
	}

	gsl_lite::span<const IndexType> col_indices() const {
		return gsl_lite::make_span(col_indices_);
	}

	gsl_lite::span<IndexType> row_ptrs() {
		return gsl_lite::make_span(row_ptrs_);
	}

	gsl_lite::span<const IndexType> row_ptrs() const {
		return gsl_lite::make_span(row_ptrs_);
	}

	size_t num_rows() const {
		return row_ptrs_.size() - 1;
	}

	size_t num_cols() const {
		return num_cols_;
	}

   private:
	std::pmr::vector<DataType> values_;
	std::pmr::vector<IndexType> col_indices_;
	std::pmr::vector<IndexType> row_ptrs_;
	size_t num_cols_;
};

template <typename DataType, typename IndexType>
class CSR_view {

   public:
	template <typename OtherDataType, typename OtherIndexType>
	CSR_view(CSR<OtherDataType, OtherIndexType>& owner)
	    : values(owner.values()),
	      col_indices(owner.col_indices()),
	      row_ptrs(owner.row_ptrs()),
	      num_cols(owner.num_cols()) {
	}

	template <typename OtherDataType, typename OtherIndexType>
	CSR_view(const CSR<OtherDataType, OtherIndexType>& owner)
	    : values(owner.values()),
	      col_indices(owner.col_indices()),
	      row_ptrs(owner.row_ptrs()),
	      num_cols(owner.num_cols()) {
	}

	CSR_view(const CSR_view& other) = default;

	gsl_lite::span<DataType> values;
	gsl_lite::span<IndexType> col_indices;
	gsl_lite::span<IndexType> row_ptrs;
	size_t num_cols;
};

} // namespace thrustshift
