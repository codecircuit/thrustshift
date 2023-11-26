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

#include <thrustshift/copy.h>
#include <thrustshift/defines.h>
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

	COO()
	    : values_(0, &pmr::default_resource),
	      row_indices_(0, &pmr::default_resource),
	      col_indices_(0, &pmr::default_resource),
	      num_cols_(0),
	      num_rows_(0) {
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

	THRUSTSHIFT_FHD gsl_lite::span<DataType> values() {
		return values_;
	}

	THRUSTSHIFT_FHD gsl_lite::span<const DataType> values() const {
		return values_;
	}

	THRUSTSHIFT_FHD gsl_lite::span<IndexType> row_indices() {
		return row_indices_;
	}

	THRUSTSHIFT_FHD gsl_lite::span<const IndexType> row_indices() const {
		return row_indices_;
	}

	THRUSTSHIFT_FHD gsl_lite::span<IndexType> col_indices() {
		return col_indices_;
	}

	THRUSTSHIFT_FHD gsl_lite::span<const IndexType> col_indices() const {
		return col_indices_;
	}

	THRUSTSHIFT_FHD size_t num_rows() const {
		return num_rows_;
	}

	THRUSTSHIFT_FHD size_t num_cols() const {
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
void diagmm(cudaStream_t& stream,
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
	constexpr unsigned block_dim = 128;
	const unsigned grid_dim = ceil_divide(nnz, block_dim);
	kernel::diagmm<RangeT, COO_T0, COO_I0, COO_T1, COO_I1>
	    <<<grid_dim, block_dim, 0, stream>>>(diag_mtx, coo_mtx, result_coo_mtx);
	THRUSTSHIFT_CHECK_CUDA_ERROR(cudaGetLastError());
}

} // namespace async

namespace kernel {

template <typename T0, typename I, typename T1>
__global__ void get_diagonal(COO_view<const T0, const I> mtx,
                             gsl_lite::span<T1> diag) {

	const auto gtid = threadIdx.x + blockIdx.x * blockDim.x;
	const auto values = mtx.values();
	const auto nnz = values.size();
	if (gtid < nnz) {
		const auto row_id = mtx.row_indices()[gtid];
		const auto col_id = mtx.col_indices()[gtid];
		if (row_id == col_id) {
			diag[row_id] = values[gtid];
		}
	}
}

template <typename T0, typename I, typename T1>
__global__ void fill_diagonal(COO_view<T0, const I> mtx, T1 value) {

	const auto gtid = threadIdx.x + blockIdx.x * blockDim.x;
	const auto values = mtx.values();
	const auto nnz = values.size();
	if (gtid < nnz) {
		const auto row_id = mtx.row_indices()[gtid];
		const auto col_id = mtx.col_indices()[gtid];
		if (row_id == col_id) {
			values[gtid] = value;
		}
	}
}

} // namespace kernel

namespace async {

template <typename T, class COO>
void get_diagonal(cudaStream_t& stream, COO&& mtx, gsl_lite::span<T> diag) {

	gsl_Expects(diag.size() == std::min(mtx.num_rows(), mtx.num_cols()));
	using I = typename std::remove_reference<COO>::type::index_type;
	using T0 = typename std::remove_reference<COO>::type::value_type;
	const auto nnz = mtx.values().size();
	constexpr unsigned block_dim = 128;
	const unsigned grid_dim = ceil_divide(
	    gsl_lite::narrow<unsigned>(nnz), gsl_lite::narrow<unsigned>(block_dim));
	fill(stream, diag, 0);
	COO_view<const T0, const I> mtx_view(mtx);
	if (nnz != 0) {
		kernel::get_diagonal<T0, I, T>
		    <<<grid_dim, block_dim, 0, stream>>>(mtx_view, diag);
		THRUSTSHIFT_CHECK_CUDA_ERROR(cudaGetLastError());
	}
}

template <typename T, class COO>
void fill_diagonal(cudaStream_t& stream, COO&& mtx, T value) {

	using I = typename std::remove_reference<COO>::type::index_type;
	using T0 = typename std::remove_reference<COO>::type::value_type;
	const auto nnz = mtx.values().size();
	constexpr unsigned block_dim = 128;
	const unsigned grid_dim = ceil_divide(
	    gsl_lite::narrow<unsigned>(nnz), gsl_lite::narrow<unsigned>(block_dim));
	if (nnz != 0) {
		kernel::fill_diagonal<T0, I, T>
		    <<<grid_dim, block_dim, 0, stream>>>(mtx, value);
		THRUSTSHIFT_CHECK_CUDA_ERROR(cudaGetLastError());
	}
}

} // namespace async

/* \brief Transform two sparse COO matrices coefficient-wise `op(op_a(a), op_b(b))`.
 * \param a first sparse COO matrix.
 * \param b second sparse COO matrix.
 * \param op Binary operator how coefficients at the same positions are combined.
 * \param op_a Unary operator which is applied to all non-zero coefficients of `a`.
 * \param op_b Unary operator which is applied to all non-zero coefficients of `b`.
 * \return COO matrix allocated with `memory_resource`
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

	if constexpr (sizeof(IndexType) == 4) {
		using KeyT = uint64_t;
		auto [tmp0, values] = make_not_a_vector_and_span<DataType>(
		    nnz_a + nnz_b, memory_resource);
		auto [tmp1, row_and_col_indices] =
		    make_not_a_vector_and_span<KeyT>(nnz_a + nnz_b, memory_resource);
		auto row_and_col_indices_ptr = row_and_col_indices.data();

		auto row_indices_a_ptr = a.row_indices().data();
		auto col_indices_a_ptr = a.col_indices().data();
		auto row_indices_b_ptr = b.row_indices().data();
		auto col_indices_b_ptr = b.col_indices().data();

		cudaStream_t stream = 0;

		async::transform(stream, a.values(), values.first(nnz_a), op_a);
		auto cit = thrust::make_counting_iterator(IndexType(0));
		//
		// Pack
		//
		thrust::for_each(thrust::cuda::par.on(stream),
		                 cit,
		                 cit + nnz_a,
		                 [row_and_col_indices_ptr,
		                  row_indices_a_ptr,
		                  col_indices_a_ptr] __host__ __device__(IndexType i) {
			                 const uint64_t fst = uint64_t(row_indices_a_ptr[i])
			                                      << 32;
			                 const uint64_t snd = col_indices_a_ptr[i];
			                 const uint64_t k = fst + snd;
			                 row_and_col_indices_ptr[i] = k;
		                 });
		thrust::for_each(thrust::cuda::par.on(stream),
		                 cit,
		                 cit + nnz_b,
		                 [row_and_col_indices_ptr,
		                  row_indices_b_ptr,
		                  col_indices_b_ptr,
		                  nnz_a] __host__ __device__(IndexType i) {
			                 const uint64_t fst = uint64_t(row_indices_b_ptr[i])
			                                      << 32;
			                 const uint64_t snd = col_indices_b_ptr[i];
			                 const uint64_t k = fst + snd;
			                 row_and_col_indices_ptr[i + nnz_a] = k;
		                 });

		async::transform(stream, b.values(), values.subspan(nnz_a), op_b);

		auto keys_begin = row_and_col_indices.data();
		auto keys_end = keys_begin + nnz_a + nnz_b;

		std::pmr::polymorphic_allocator<KeyT> alloc(&memory_resource);
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
		COO<DataType, IndexType> result(
		    nnz_result, num_rows, num_cols, memory_resource);
		auto [tmp2, key_result] =
		    thrustshift::make_not_a_vector_and_span<uint64_t>(nnz_result,
		                                                      memory_resource);
		auto keys_result_begin = key_result.data();
		thrust::reduce_by_key(thrust::cuda::par(alloc),
		                      keys_begin,
		                      keys_end,
		                      values.begin(),
		                      keys_result_begin,
		                      result.values().begin(),
		                      thrust::equal_to<KeyT>(),
		                      op);
		//
		// Unpack
		//
		auto result_row_indices_ptr = result.row_indices().data();
		auto result_col_indices_ptr = result.col_indices().data();
		thrust::for_each(
		    thrust::cuda::par.on(stream),
		    cit,
		    cit + nnz_result,
		    [keys_result_begin,
		     result_row_indices_ptr,
		     result_col_indices_ptr] __host__ __device__(IndexType i) {
			    const uint64_t key = keys_result_begin[i];
			    const uint64_t fst = key >> 32;
			    const uint64_t snd = (key << 32) >> 32;
			    result_row_indices_ptr[i] = fst;
			    result_col_indices_ptr[i] = snd;
		    });
		result.set_storage_order(storage_order_t::row_major);
		return result;
	}
	else {
		auto [tmp0, values] = make_not_a_vector_and_span<DataType>(
		    nnz_a + nnz_b, memory_resource);
		auto [tmp1, row_indices] = make_not_a_vector_and_span<IndexType>(
		    nnz_a + nnz_b, memory_resource);
		auto [tmp2, col_indices] = make_not_a_vector_and_span<IndexType>(
		    nnz_a + nnz_b, memory_resource);

		cudaStream_t stream = 0;

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

		std::pmr::polymorphic_allocator<KeyT> alloc(&memory_resource);
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
		COO<DataType, IndexType> result(
		    nnz_result, num_rows, num_cols, memory_resource);
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
}

template <typename DataType,
          typename IndexType,
          class BinaryOperator,
          class UnaryOperatorA0,
          class UnaryOperatorA1,
          class UnaryOperatorB0,
          class UnaryOperatorB1,
          class MemoryResource>
std::tuple<thrustshift::COO<DataType, IndexType>,
           decltype(thrustshift::make_not_a_vector<DataType>(1,
                                                             MemoryResource{}))>
transform2(thrustshift::COO_view<const DataType, const IndexType> a,
           thrustshift::COO_view<const DataType, const IndexType> b,
           BinaryOperator&& op,
           UnaryOperatorA0&& op_a0,
           UnaryOperatorA1&& op_a1,
           UnaryOperatorB0&& op_b0,
           UnaryOperatorB1&& op_b1,
           MemoryResource& memory_resource) {

	const auto nnz_a = a.values().size();
	const auto nnz_b = b.values().size();
	const auto num_rows = a.num_rows();
	const auto num_cols = b.num_cols();
	gsl_Expects(b.num_rows() == num_rows);
	gsl_Expects(b.num_cols() == num_cols);

	if constexpr (sizeof(IndexType) == 4) {
		using KeyT = uint64_t;
		auto [tmp0, values0] = make_not_a_vector_and_span<DataType>(
		    nnz_a + nnz_b, memory_resource);
		auto [tmp4, values1] = make_not_a_vector_and_span<DataType>(
		    nnz_a + nnz_b, memory_resource);
		auto [tmp1, row_and_col_indices] =
		    make_not_a_vector_and_span<KeyT>(nnz_a + nnz_b, memory_resource);
		auto row_and_col_indices_ptr = row_and_col_indices.data();

		auto row_indices_a_ptr = a.row_indices().data();
		auto col_indices_a_ptr = a.col_indices().data();
		auto row_indices_b_ptr = b.row_indices().data();
		auto col_indices_b_ptr = b.col_indices().data();

		cudaStream_t stream = 0;

		async::transform(stream, a.values(), values0.first(nnz_a), op_a0);
		async::transform(stream, b.values(), values0.subspan(nnz_a), op_b0);
		async::transform(stream, a.values(), values1.first(nnz_a), op_a1);
		async::transform(stream, b.values(), values1.subspan(nnz_a), op_b1);

		auto cit = thrust::make_counting_iterator(IndexType(0));
		//
		// Pack
		//
		thrust::for_each(thrust::cuda::par.on(stream),
		                 cit,
		                 cit + nnz_a,
		                 [row_and_col_indices_ptr,
		                  row_indices_a_ptr,
		                  col_indices_a_ptr] __host__ __device__(IndexType i) {
			                 const uint64_t fst = uint64_t(row_indices_a_ptr[i])
			                                      << 32;
			                 const uint64_t snd = col_indices_a_ptr[i];
			                 const uint64_t k = fst + snd;
			                 row_and_col_indices_ptr[i] = k;
		                 });
		thrust::for_each(thrust::cuda::par.on(stream),
		                 cit,
		                 cit + nnz_b,
		                 [row_and_col_indices_ptr,
		                  row_indices_b_ptr,
		                  col_indices_b_ptr,
		                  nnz_a] __host__ __device__(IndexType i) {
			                 const uint64_t fst = uint64_t(row_indices_b_ptr[i])
			                                      << 32;
			                 const uint64_t snd = col_indices_b_ptr[i];
			                 const uint64_t k = fst + snd;
			                 row_and_col_indices_ptr[i + nnz_a] = k;
		                 });

		auto keys_begin = row_and_col_indices.data();
		auto keys_end = keys_begin + nnz_a + nnz_b;

		auto zip_value_it = thrust::make_zip_iterator(
		    thrust::make_tuple(values0.begin(), values1.begin()));

		std::pmr::polymorphic_allocator<KeyT> alloc(&memory_resource);
		thrust::sort_by_key(thrust::cuda::par(alloc),
		                    keys_begin,
		                    keys_end,
		                    zip_value_it.begin());

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
		COO<DataType, IndexType> result(
		    nnz_result, num_rows, num_cols, memory_resource);

		auto result_values0 = result.values();
		auto [result_values1_nav, result_values1] =
		    thrustshift::make_not_a_vector_and_span<DataType>(nnz_result,
		                                                      memory_resource);
		auto [tmp2, key_result] =
		    thrustshift::make_not_a_vector_and_span<uint64_t>(nnz_result,
		                                                      memory_resource);
		auto keys_result_begin = key_result.data();

		auto zip_result_it = thrust::make_zip_iterator(
		    thrust::make_tuple(result_values0.data(), result_values1.data()));

		auto opp = [op] __host__ __device__(
		               const thrust::tuple<DataType, DataType>& tup0,
		               const thrust::tuple<DataType, DataType>& tup1) {
			return thrust::make_tuple(
			    op(thrust::get<0>(tup0), thrust::get<0>(tup1)),
			    op(thrust::get<1>(tup0), thrust::get<1>(tup1)));
		};
		thrust::reduce_by_key(thrust::cuda::par(alloc),
		                      keys_begin,
		                      keys_end,
		                      zip_value_it.begin(),
		                      keys_result_begin,
		                      zip_result_it,
		                      thrust::equal_to<KeyT>(),
		                      opp);
		//
		// Unpack
		//
		auto result_row_indices_ptr = result.row_indices().data();
		auto result_col_indices_ptr = result.col_indices().data();
		thrust::for_each(
		    thrust::cuda::par.on(stream),
		    cit,
		    cit + nnz_result,
		    [keys_result_begin,
		     result_row_indices_ptr,
		     result_col_indices_ptr] __host__ __device__(IndexType i) {
			    const uint64_t key = keys_result_begin[i];
			    const uint64_t fst = key >> 32;
			    const uint64_t snd = (key << 32) >> 32;
			    result_row_indices_ptr[i] = fst;
			    result_col_indices_ptr[i] = snd;
		    });
		result.set_storage_order(storage_order_t::row_major);
		return std::make_tuple(std::move(result),
		                       std::move(result_values1_nav));
	}
}

template <typename DataType, typename IndexType, class MemoryResource>
COO<DataType, IndexType> symmetrize_abs(
    COO_view<const DataType, const IndexType> mtx,
    MemoryResource& memory_resource) {

	if (mtx.values().empty()) {
		return COO<DataType, IndexType>(
		    0, mtx.num_rows(), mtx.num_cols(), memory_resource);
	}

	// transposed view
	COO_view<const DataType, const IndexType> mtx_trans(mtx.values(),
	                                                    mtx.col_indices(),
	                                                    mtx.row_indices(),
	                                                    mtx.num_rows(),
	                                                    mtx.num_cols());

	auto abs = [] __host__ __device__(DataType x) { return std::abs(x); };

	return transform(
	    mtx,
	    mtx_trans,
	    [] __host__ __device__(DataType x, DataType y) { return x + y; },
	    abs,
	    abs,
	    memory_resource);
}

template <typename DataType, typename IndexType, class MemoryResource>
COO<DataType, IndexType> make_pattern_symmetric(
    COO_view<const DataType, const IndexType> mtx,
    MemoryResource& memory_resource) {

	if (mtx.values().empty()) {
		return COO<DataType, IndexType>(
		    0, mtx.num_rows(), mtx.num_cols(), memory_resource);
	}

	COO_view<const DataType, const IndexType> mtx_trans(mtx.values(),
	                                                    mtx.col_indices(),
	                                                    mtx.row_indices(),
	                                                    mtx.num_rows(),
	                                                    mtx.num_cols());

	auto identity = [] __host__ __device__(DataType x) { return x; };
	auto make_zero = [] __host__ __device__(DataType x) { return DataType(0); };

	return transform(
	    mtx,
	    mtx_trans,
	    [] __host__ __device__(DataType x, DataType y) { return x + y; },
	    identity,
	    make_zero,
	    memory_resource);
}

template <typename DataType, typename IndexType, class MemoryResource>
std::tuple<thrustshift::COO<DataType, IndexType>,
           decltype(thrustshift::make_not_a_vector<DataType>(1,
                                                             MemoryResource{}))>
symmetrize_abs_and_make_pattern_symmetric(
    COO_view<const DataType, const IndexType> mtx,
    MemoryResource& memory_resource) {

	gsl_Expects(!mtx.values().empty());

	COO_view<const DataType, const IndexType> mtx_trans(mtx.values(),
	                                                    mtx.col_indices(),
	                                                    mtx.row_indices(),
	                                                    mtx.num_rows(),
	                                                    mtx.num_cols());
	auto identity = [] __host__ __device__(DataType x) { return x; };
	auto make_zero = [] __host__ __device__(DataType x) { return DataType(0); };
	auto abs = [] __host__ __device__(DataType x) { return std::abs(x); };

	return transform2(
	    mtx,
	    mtx_trans,
	    [] __host__ __device__(DataType x, DataType y) { return x + y; },
	    abs,
	    identity,
	    abs,
	    make_zero,
	    memory_resource);
}

} // namespace thrustshift
