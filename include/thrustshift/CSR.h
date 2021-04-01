#pragma once

#include <memory_resource>
#include <vector>

#include <gsl-lite/gsl-lite.hpp>

#include <thrust/sort.h>
#include <cuda/define_specifiers.hpp>

#include <thrustshift/memory-resource.h>

namespace thrustshift {

//! Col indices must be ordered
template <typename DataType, typename IndexType>
class CSR {
   public:
	using value_type = DataType;
	using index_type = IndexType;

   private:
	bool cols_are_sorted() {
		for (size_t row_id = 0; row_id < this->num_rows(); ++row_id) {
			if (!std::is_sorted(col_indices_.begin() + row_ptrs_[row_id],
			                    col_indices_.begin() + row_ptrs_[row_id + 1])) {
				return false;
			}
		}
		return true;
	}

   public:
	CSR() : row_ptrs_(1, 0, &pmr::default_resource), num_cols_(0) {
	}

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
		gsl_ExpectsAudit(cols_are_sorted());
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
	          pmr::default_resource) {
	}

	// The copy constructor is declared explicitly to ensure
	// managed memory is used per default.
	CSR(const CSR& other)
	    : CSR(other.values(),
	          other.col_indices(),
	          other.row_ptrs(),
	          other.num_cols()) {
	}

	CSR(CSR&& other) = default;

	CSR& operator=(const CSR& other) = default;

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

	void sort_column_indices() {
		for (size_t row_id = 0; row_id < num_rows(); ++row_id) {
			const auto nns_start = row_ptrs_[row_id];
			const auto nns_end = row_ptrs_[row_id + 1];
			thrust::sort_by_key(thrust::host,
			                    col_indices_.begin() + nns_start,
			                    col_indices_.begin() + nns_end,
			                    values_.begin() + nns_start);
		}
	}

	/*! \brief Add additional elements to each row of the matrix.
	 *
	 *  A row might already have the maximum number of element. Therefore
	 *  not every row might be extended by exactly `num_max_additional_elements_per_row`.
	 *
	 */
	void extend_rows(int num_max_additional_elements_per_row,
	                 DataType value = 0) {
		if (num_max_additional_elements_per_row == 0) {
			return;
		}
		gsl_Expects(num_max_additional_elements_per_row >= 0);
		std::vector<DataType> tmp_values(values_.begin(), values_.end());
		std::vector<IndexType> tmp_col_indices(col_indices_.begin(),
		                                       col_indices_.end());
		size_t num_additional_elements = 0;
		const int nc = gsl_lite::narrow<int>(num_cols());
		auto get_num_additional_elements_per_row = [&](size_t row_id) {
			const int num_elements_curr_row =
			    row_ptrs_[row_id + 1] - row_ptrs_[row_id];
			return std::max(std::min(num_max_additional_elements_per_row,
			                         nc - num_elements_curr_row),
			                0);
		};
		for (size_t row_id = 0; row_id < num_rows(); ++row_id) {
			num_additional_elements +=
			    get_num_additional_elements_per_row(row_id);
		}

		const size_t new_nnz = values_.size() + num_additional_elements;
		values_.resize(new_nnz);
		col_indices_.resize(new_nnz);
		int nns_offset = 0;
		for (size_t row_id = 0; row_id < num_rows(); ++row_id) {
			const int num_additional_elements_curr_row =
			    get_num_additional_elements_per_row(row_id);
			int nns_id = row_ptrs_[row_id];
			const int nns_end = row_ptrs_[row_id + 1];
			const int nnz_curr_row = nns_end - nns_id;
			int curr_col_id = 0;
			int num_added_elements = 0;
			for (int new_nns_id = nns_offset;
			     new_nns_id < nns_offset + num_additional_elements_curr_row +
			                      row_ptrs_[row_id + 1] - row_ptrs_[row_id];
			     ++new_nns_id) {
				const int other_col_id = nns_id == nns_end
				                             ? std::numeric_limits<int>::max()
				                             : tmp_col_indices[nns_id];
				if (curr_col_id < other_col_id &&
				    num_added_elements < num_additional_elements_curr_row) {
					values_[new_nns_id] = value;
					col_indices_[new_nns_id] = curr_col_id;
					++curr_col_id;
					++num_added_elements;
				}
				else {
					values_[new_nns_id] = tmp_values[nns_id];
					col_indices_[new_nns_id] = other_col_id;
					curr_col_id = other_col_id + 1;
					++nns_id;
				}
			}
			row_ptrs_[row_id] = nns_offset;
			nns_offset += nnz_curr_row + num_additional_elements_curr_row;
		}
		row_ptrs_.back() = values_.size();
		gsl_ExpectsAudit(std::is_sorted(row_ptrs_.begin(), row_ptrs_.end()));
		gsl_ExpectsAudit(cols_are_sorted());
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
	using value_type = DataType;
	using index_type = IndexType;

	template <typename OtherDataType, typename OtherIndexType>
	CSR_view(CSR<OtherDataType, OtherIndexType>& owner)
	    : values_(owner.values()),
	      col_indices_(owner.col_indices()),
	      row_ptrs_(owner.row_ptrs()),
	      num_cols_(owner.num_cols()) {
	}

	template <typename OtherDataType, typename OtherIndexType>
	CSR_view(const CSR<OtherDataType, OtherIndexType>& owner)
	    : values_(owner.values()),
	      col_indices_(owner.col_indices()),
	      row_ptrs_(owner.row_ptrs()),
	      num_cols_(owner.num_cols()) {
	}

	template <class DataRange,
	          class ColIndRange,
	          class RowPtrsRange>
	CSR_view(DataRange&& values,
	    ColIndRange&& col_indices,
	    RowPtrsRange&& row_ptrs,
	    size_t num_cols)
	    : values_(values),
	      col_indices_(col_indices),
	      row_ptrs_(row_ptrs),
	      num_cols_(num_cols) {
		gsl_Expects(values.size() == col_indices.size());
		gsl_Expects(row_ptrs.size() > 0);
		gsl_ExpectsAudit(row_ptrs[0] == 0);
	}

	CSR_view(const CSR_view& other) = default;
	CSR_view(CSR_view&& other) = default;

	CUDA_FHD gsl_lite::span<DataType> values() {
		return values_;
	}

	CUDA_FHD gsl_lite::span<const DataType> values() const {
		return values_;
	}

	CUDA_FHD gsl_lite::span<IndexType> col_indices() {
		return col_indices_;
	}

	CUDA_FHD gsl_lite::span<const IndexType> col_indices() const {
		return col_indices_;
	}

	CUDA_FHD gsl_lite::span<IndexType> row_ptrs() {
		return row_ptrs_;
	}

	CUDA_FHD gsl_lite::span<const IndexType> row_ptrs() const {
		return row_ptrs_;
	}

	CUDA_FHD size_t num_rows() const {
		return row_ptrs_.size() - 1;
	}

	CUDA_FHD size_t num_cols() const {
		return num_cols_;
	}

	CUDA_FHD IndexType max_row_nnz() const {
		IndexType mnnz = 0;
		for (size_t row_id = 1; row_id < row_ptrs_.size(); ++row_id) {
			mnnz = std::max(row_ptrs_[row_id] - row_ptrs_[row_id - 1], mnnz);
		}
		return mnnz;
	}

   private:
	gsl_lite::span<DataType> values_;
	gsl_lite::span<IndexType> col_indices_;
	gsl_lite::span<IndexType> row_ptrs_;
	size_t num_cols_;
};

} // namespace thrustshift
