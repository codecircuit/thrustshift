#pragma once

#include <cstddef>
#include <iostream>
#include <limits>

#include <gsl-lite/gsl-lite.hpp>

#include <cub/cub.cuh>

#include <thrustshift/defines.h>
#include <thrustshift/fill.h>
#include <thrustshift/math.h>
#include <thrustshift/not-a-vector.h>

namespace thrustshift {

namespace kernel {

template <typename IndexT>
__global__ void wrap_subgroups(gsl_lite::span<const IndexT> subgroup_ptrs,
                               gsl_lite::span<IndexT> group_ptrs,
                               IndexT mean_group_size) {
	const IndexT gtid = threadIdx.x + blockIdx.x * blockDim.x;
	const IndexT num_rows = subgroup_ptrs.size() - 1;
	const IndexT N = group_ptrs.size() - 1;
	if (gtid < num_rows) {
		const IndexT row_id = gtid;
		const IndexT curr_work = subgroup_ptrs[row_id + 1];
		const IndexT prev_work = subgroup_ptrs[row_id];
		const IndexT l = curr_work / mean_group_size;
		const IndexT m = prev_work / mean_group_size;
		if (prev_work != curr_work && l != 0 && (l != m) &&
		    l != N) { // only first thread is allowed to write to last element
			group_ptrs[l] = row_id + 1;
		}
	}
	if (gtid == 0) {
		group_ptrs[N] = num_rows;
		group_ptrs[0] = 0;
	}
}

} // namespace kernel

namespace {

template <typename StorageIndex>
struct unequal_to_functor {
	StorageIndex x;
	THRUSTSHIFT_FD bool operator()(StorageIndex y) const {
		return y != x;
	}
};

} // namespace

namespace async {

/*! \brief Wrap subsequent subgroups of varying size into groups.
 *
 *  No subgroup can be part of two groups. Each subgroup can
 *  contain a certain amount of elements (also no elements at all). This
 *  function tries to wrap the subgroups into bigger groups of size `mean_group_size`.
 *  This function is especially useful to distribute the work amongst CUDA thread blocks,
 *  which operate on rows of a sparse matrix. If a single subgroup is larger than `mean_group_size`
 *  the subgroup will form a group on its own, which is larger than `mean_group_size`.
 *  The last group is rather larger than `mean_group_size` instead of creating one
 *  group which is smaller than `mean_group_size`.
 *
 *  \param stream CUDA stream.
 *  \param subgroup_ptrs Array of length `num_subgroups + 1`. `subgroup_ptrs[i]` denotes
 *    the index of the first element of subgroup `i`. `subgroup_ptrs[num_subgroups]` must
 *    be equal to `num_elements`.
 *  \param num_elements The sum of the sizes of all subgroups.
 *  \param group_ptrs The result. `group_ptrs[k]` is the start subgroup ID of group `i`, whereas
 *    `group_ptrs[k+1]` is the end subgroup ID of group `i`. Thus group `i` contains all subgroups
 *    in range `[group_ptrs[k], group_ptrs[k+1])`. The last valid output
 *    is `group_ptrs[*group_ptrs_size - 1] = subgroup_ptrs.size() - 1`. The maximum number of groups
 *    is `num_elements / mean_group_size + 1`. The minimum number of groups is 2. Thus the buffer
 *    of `group_ptrs` must be of size `max(num_elements / mean_group_size + 1, 2)`.
 *  \param mean_group_size Intended group size, which is not always possible. E.g. if there
 *    is only one subgroup, there can also be only one group.
 *  \param group_ptrs_size Pointer to an element in GPU memory. After successful execution the
 *    total number of groups is written to this variable.
 */
template <typename IndexT, class MemoryResource>
void wrap_subgroups(cudaStream_t& stream,
                    gsl_lite::span<const IndexT> subgroup_ptrs,
                    IndexT num_elements,
                    gsl_lite::span<IndexT> group_ptrs,
                    IndexT mean_group_size,
                    gsl_lite::not_null<IndexT*> group_ptrs_size,
                    MemoryResource& delayed_memory_resource) {

	constexpr unsigned block_dim = 128;

	if (subgroup_ptrs.size() < 2) {
		// There is not a single subgroup
		return;
	}

	if (gsl_lite::narrow<IndexT>(group_ptrs.size()) !=
	    std::max(num_elements / mean_group_size + 1, (IndexT) 2)) {
		std::cerr << "ERROR: group_ptrs buffer has wrong size! actual "
		             "size = "
		          << group_ptrs.size()
		          << ", required size = max(num_elements / "
		             "mean_group_size + 1, 2) = "
		          << std::max(num_elements / mean_group_size + 1, 2)
		          << std::endl;
		std::terminate();
	}

	unequal_to_functor<IndexT> f({std::numeric_limits<IndexT>::max()});

	gsl_lite::span<std::byte> tmp_storage;
	size_t tmp_storage_size = 0;

	auto enqueue_select_if = [&]() {
		THRUSTSHIFT_CHECK_CUDA_ERROR(cub::DeviceSelect::If(tmp_storage.data(),
		                                                   tmp_storage_size,
		                                                   group_ptrs.data(),
		                                                   group_ptrs.data(),
		                                                   group_ptrs_size,
		                                                   group_ptrs.size(),
		                                                   f,
		                                                   stream));
	};
	// set temporary memory size only
	enqueue_select_if();

	auto tmp =
	    make_not_a_vector<std::byte>(tmp_storage_size, delayed_memory_resource);
	tmp_storage = tmp.to_span();

	fill(stream, group_ptrs, std::numeric_limits<IndexT>::max());

	gsl_Expects(subgroup_ptrs.size() > 0);
	const IndexT num_rows = subgroup_ptrs.size() - 1;
	const IndexT grid_dim =
	    ceil_divide(num_rows, gsl_lite::narrow<IndexT>(block_dim));
	kernel::wrap_subgroups<IndexT><<<grid_dim, block_dim, 0, stream>>>(
	    subgroup_ptrs, group_ptrs, mean_group_size);
	THRUSTSHIFT_CHECK_CUDA_ERROR(cudaGetLastError());
	enqueue_select_if();
}

} // namespace async

} // namespace thrustshift
