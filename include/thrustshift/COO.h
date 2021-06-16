#pragma once

#include <memory_resource>
#include <vector>

#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/inner_product.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/merge.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/tuple.h>

#include <gsl-lite/gsl-lite.hpp>

#include <cuda/define_specifiers.hpp>

#include <thrustshift/copy.h>
#include <thrustshift/equal.h>
#include <thrustshift/managed-vector.h>
#include <thrustshift/math.h>
#include <thrustshift/memory-resource.h>
#include <thrustshift/not-a-vector.h>
#include <thrustshift/transform.h>

namespace thrustshift {

enum class storage_order_t { row_major, col_major, none };

template <typename DataType, typename IndexType>
class COO {

   public:
	using value_type = DataType;
	using index_type = IndexType;

	COO() : num_cols_(0), num_rows_(0) {
	}

	template <class MemoryResource>
	COO(size_t nnz,
	    size_t num_rows,
	    size_t num_cols,
	    MemoryResource& memory_resource)
	    : values_(nnz, &memory_resource),
	      row_indices_(nnz, &memory_resource),
	      col_indices_(nnz, &memory_resource),
	      num_rows_(num_rows),
	      num_cols_(num_cols),
	      storage_order_(storage_order_t::none) {
	}

	COO(size_t nnz, size_t num_rows, size_t num_cols)
	    : COO(nnz, num_rows, num_cols, pmr::default_resource) {
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
	    storage_order_t storage_order,
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
	      storage_order_(storage_order) {
		gsl_Expects(values.size() == row_indices.size());
		gsl_Expects(values.size() == col_indices.size());
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
	    : COO(std::forward<DataRange>(values),
	          std::forward<RowIndRange>(row_indices),
	          std::forward<ColIndRange>(col_indices),
	          num_rows,
	          num_cols,
	          storage_order_t::none,
	          memory_resource) {
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
	          pmr::default_resource) {
	}

	// The copy constructor is declared explicitly to ensure
	// managed memory is used per default.
	COO(const COO& other)
	    : COO(other.values(),
	          other.row_indices(),
	          other.col_indices(),
	          other.num_rows(),
	          other.num_cols(),
	          other.get_storage_order(),
	          pmr::default_resource) {
	}

	COO(COO&& other) = default;

	void change_storage_order(storage_order_t new_storage_order) {
		IndexType* primary_keys_first;
		IndexType* secondary_keys_first;

		if (new_storage_order == storage_order_) {
			return;
		}

		switch (new_storage_order) {
			case storage_order_t::row_major:
				primary_keys_first = row_indices_.data();
				secondary_keys_first = col_indices_.data();
				break;
			case storage_order_t::col_major:
				primary_keys_first = col_indices_.data();
				secondary_keys_first = row_indices_.data();
				break;
			case storage_order_t::none:
				return;
		}

		// Thrust's relational operators are overloaded for thrust::pair.
		// This ensures that we sort with respect to the second key if the first key is equal.
		auto key_it = thrust::make_zip_iterator(
		    thrust::make_tuple(primary_keys_first, secondary_keys_first));
		thrust::sort_by_key(
		    thrust::cuda::par, key_it, key_it + values_.size(), values_.data());
		storage_order_ = new_storage_order;
	}

	// If the storage order is changed externally
	void set_storage_order(storage_order_t new_storage_order) {
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
			for (int nns_id = gsl_lite::narrow<int>(indices.size()) - 1;
			     nns_id >= 0;
			     --nns_id) {
				ptrs[indices[nns_id]] = nns_id;
			}
			ptrs[0] = 0;
			ptrs.back() = indices.size();
			index_type k = indices.size();
			for (int id = gsl_lite::narrow<int>(ptrs.size()) - 1; id > 0;
			     --id) {
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

//! Storage order does not affect comparison operator. Therefore this operator is defined explicitly.
template <typename DataType, typename IndexType>
bool operator==(const COO<DataType, IndexType>& a,
                const COO<DataType, IndexType>& b) {
	return equal(a.values(), b.values()) &&
	       equal(a.row_indices(), b.row_indices()) &&
	       equal(a.col_indices(), b.col_indices()) &&
	       a.num_rows() == b.num_rows() && a.num_cols() == b.num_cols();
}

template <typename DataType, typename IndexType>
class COO_view {

   public:
	using value_type = DataType;
	using index_type = IndexType;

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

	template <class ValueRange, class ColIndRange, class RowIndRange>
	COO_view(ValueRange&& values,
	         RowIndRange&& row_indices,
	         ColIndRange&& col_indices,
	         size_t num_rows,
	         size_t num_cols)
	    : values_(values),
	      row_indices_(row_indices),
	      col_indices_(col_indices),
	      num_rows_(num_rows),
	      num_cols_(num_cols) {
		const auto nnz = values.size();
		gsl_Expects(col_indices.size() == nnz);
		gsl_Expects(row_indices.size() == nnz);
	}

	COO_view(const COO_view& other) = default;

	CUDA_FHD gsl_lite::span<DataType> values() {
		return values_;
	}

	CUDA_FHD gsl_lite::span<const DataType> values() const {
		return values_;
	}

	CUDA_FHD gsl_lite::span<IndexType> row_indices() {
		return row_indices_;
	}

	CUDA_FHD gsl_lite::span<const IndexType> row_indices() const {
		return row_indices_;
	}

	CUDA_FHD gsl_lite::span<IndexType> col_indices() {
		return col_indices_;
	}

	CUDA_FHD gsl_lite::span<const IndexType> col_indices() const {
		return col_indices_;
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

namespace kernel {

template <typename T, typename T0, typename I0, typename T1, typename I1>
__global__ void diagmm(gsl_lite::span<const T> diag,
                       COO_view<const T0, const I0> mtx,
                       COO_view<T1, I1> result_mtx) {

	const auto nnz = mtx.values().size();
	const auto gtid = threadIdx.x + blockIdx.x * blockDim.x;
	if (gtid < nnz) {
		result_mtx.values()[gtid] =
		    diag[mtx.row_indices()[gtid]] * mtx.values()[gtid];
	}
}

} // namespace kernel

namespace async {

//! Calculate the product `result_coo_mtx = (diag_mtx * coo_mtx)`
//! result_coo_mtx can be equal to coo_mtx. Then the multiplication is in place.
template <class Range, class COO_C0, class COO_C1>
void diagmm(cuda::stream_t& stream,
            Range&& diag_mtx,
            COO_C0&& coo_mtx,
            COO_C1&& result_coo_mtx) {

	gsl_Expects(coo_mtx.values().size() == result_coo_mtx.values().size());
	gsl_Expects(coo_mtx.num_cols() == result_coo_mtx.num_cols());
	gsl_Expects(coo_mtx.num_rows() == result_coo_mtx.num_rows());
	gsl_Expects(diag_mtx.size() == coo_mtx.num_rows());
	// Only square matrices are allowed
	gsl_Expects(coo_mtx.num_rows() == coo_mtx.num_cols());

	using RangeT = typename std::remove_reference<Range>::type::value_type;
	using COO_T0 = typename std::remove_reference<COO_C0>::type::value_type;
	using COO_I0 = typename std::remove_reference<COO_C0>::type::index_type;
	using COO_T1 = typename std::remove_reference<COO_C1>::type::value_type;
	using COO_I1 = typename std::remove_reference<COO_C1>::type::index_type;

	const auto nnz = coo_mtx.values().size();
	constexpr cuda::grid::block_dimension_t block_dim = 128;
	const cuda::grid::dimension_t grid_dim = ceil_divide(nnz, block_dim);
	cuda::enqueue_launch(kernel::diagmm<RangeT, COO_T0, COO_I0, COO_T1, COO_I1>,
	                     stream,
	                     cuda::make_launch_config(grid_dim, block_dim),
	                     diag_mtx,
	                     coo_mtx,
	                     result_coo_mtx);
}

} // namespace async

namespace kernel {

template <typename T0, typename I, typename T1>
__global__ void get_diagonal(COO_view<const T0, const I> mtx,
                             gsl_lite::span<T1> diag) {

	const auto gtid = threadIdx.x + blockDim.x * blockDim.x;
	if (gtid < mtx.num_rows()) {
		const auto row_id = mtx.row_indices()[gtid];
		const auto col_id = mtx.col_indices()[gtid];
		if (row_id == col_id) {
			diag[row_id] = mtx.values()[gtid];
		}
	}
}

} // namespace kernel

namespace async {

template <typename T, class COO>
void get_diagonal(cuda::stream_t& stream, COO&& mtx, gsl_lite::span<T> diag) {

	using I = typename std::remove_reference<COO>::type::index_type;
	using T0 = typename std::remove_reference<COO>::type::value_type;
	const auto nnz = mtx.values().size();
	constexpr cuda::grid::block_dimension_t block_dim = 128;
	const cuda::grid::dimension_t grid_dim = ceil_divide(gsl_lite::narrow<cuda::grid::dimension_t>(nnz), gsl_lite::narrow<cuda::grid::dimension_t>(block_dim));
	fill(stream, diag, 0);
	cuda::enqueue_launch(kernel::get_diagonal<T0, I, T>,
	                     stream,
	                     cuda::make_launch_config(grid_dim, block_dim),
	                     mtx,
	                     diag);
}

} // namespace async

/* \brief Transform two sparse COO matrices coefficient-wise `op(op_a(a), op_b(b))`.
 * \param a first sparse COO matrix.
 * \param b second sparse COO matrix.
 * \param op Binary operator how coefficients at the same positions are combined.
 * \param op_a Unary operator which is applied to all non-zero coefficients of `a`.
 * \param op_b Unary operator which is applied to all non-zero coefficients of `b`.
 * \return COO matrix
 * \note It might be misleading that the binary operator is **not** applied to coefficients, which
 *   only appear in one of the two matrices. E.g. `op=minus`, `op_a=identity`, `op_b=identity` and `a` is a zero matrix the result
 *   is `result != a - b = -b`. Also note that it cannot be ensured that the left operand of the binary operator
 *   is always an element of `a`. Therefore the unary operators are required.
 */
template <typename DataType,
          typename IndexType,
          class BinaryOperator,
          class UnaryOperatorA,
          class UnaryOperatorB,
          class MemoryResource>
thrustshift::COO<DataType, IndexType> transform(
    thrustshift::COO_view<const DataType, const IndexType> a,
    thrustshift::COO_view<const DataType, const IndexType> b,
    BinaryOperator&& op,
    UnaryOperatorA&& op_a,
    UnaryOperatorB&& op_b,
    MemoryResource& memory_resource) {

	const auto nnz_a = a.values().size();
	const auto nnz_b = b.values().size();
	const auto num_rows = a.num_rows();
	const auto num_cols = b.num_cols();
	gsl_Expects(b.num_rows() == num_rows);
	gsl_Expects(b.num_cols() == num_cols);

	auto [tmp0, values] =
	    make_not_a_vector_and_span<DataType>(nnz_a + nnz_b, memory_resource);
	auto [tmp1, row_indices] =
	    make_not_a_vector_and_span<IndexType>(nnz_a + nnz_b, memory_resource);
	auto [tmp2, col_indices] =
	    make_not_a_vector_and_span<IndexType>(nnz_a + nnz_b, memory_resource);

	auto device = cuda::device::current::get();
	auto stream = device.default_stream();

	async::transform(stream, a.values(), values.first(nnz_a), op_a);
	async::copy(stream, a.row_indices(), row_indices.first(nnz_a));
	async::copy(stream, a.col_indices(), col_indices.first(nnz_a));

	async::transform(stream, b.values(), values.subspan(nnz_a), op_b);
	async::copy(stream, b.row_indices(), row_indices.subspan(nnz_a));
	async::copy(stream, b.col_indices(), col_indices.subspan(nnz_a));

	using KeyT = thrust::tuple<IndexType, IndexType>;
	auto keys_begin = thrust::make_zip_iterator(
	    thrust::make_tuple(row_indices.begin(), col_indices.begin()));
	auto keys_end = keys_begin + nnz_a + nnz_b;

	std::pmr::polymorphic_allocator<char> alloc(&memory_resource);
	thrust::sort_by_key(
	    thrust::cuda::par(alloc), keys_begin, keys_end, values.begin());

	const std::size_t nnz_result =
	    thrust::inner_product(thrust::cuda::par(alloc),
	                          keys_begin,
	                          keys_end - 1,
	                          keys_begin + 1,
	                          std::size_t(0),
	                          thrust::plus<std::size_t>(),
	                          thrust::not_equal_to<KeyT>()) +
	    std::size_t(1);

	// ```cpp
	//   auto memory_resource = ...;
	//   auto res = transform(..., memory_resource);
	//
	// ```
	// should work fine because the dtor of `res` is called before the dtor of `memory_resource`
	COO<DataType, IndexType> result(nnz_result, num_rows, num_cols);
	auto keys_result_begin = thrust::make_zip_iterator(thrust::make_tuple(
	    result.row_indices().begin(), result.col_indices().begin()));
	thrust::reduce_by_key(thrust::cuda::par(alloc),
	                      keys_begin,
	                      keys_end,
	                      values.begin(),
	                      keys_result_begin,
	                      result.values().begin(),
	                      thrust::equal_to<KeyT>(),
	                      op);
	result.set_storage_order(storage_order_t::row_major);
	return result;
}

template <typename DataType, typename IndexType, class MemoryResource>
COO<DataType, IndexType> symmetrize_abs(
    COO_view<const DataType, const IndexType> mtx,
    MemoryResource& memory_resource) {

	if (mtx.values().empty()) {
		return COO<DataType, IndexType>(0, mtx.num_rows(), mtx.num_cols(), memory_resource);
	}

	COO_view<const DataType, const IndexType> mtx_trans(
	    mtx.values(),
	    mtx.col_indices(),
	    mtx.row_indices(),
	    mtx.num_rows(),
	    mtx.num_cols());
	auto abs = [] __device__(DataType x) { return std::abs(x); };
	return transform(
	    mtx,
	    mtx_trans,
	    [] __device__(DataType a, DataType b) { return a + b; },
	    abs,
	    abs,
	    memory_resource);
}

} // namespace thrustshift
