#include <algorithm>
#include <bitset>
#include <numeric>
#include <variant>
#include <vector>

#include <Eigen/Sparse>

#include <gsl-lite/gsl-lite.hpp>

#include <boost/test/data/test_case.hpp>
#include <boost/test/unit_test.hpp>

#include <thrustshift/CSR.h>
#include <thrustshift/defines.h>
#include <thrustshift/managed-vector.h>
#include <thrustshift/memory-resource.h>
#include <thrustshift/random.h>
#include <thrustshift/sort.h>

namespace bdata = boost::unit_test::data;
using namespace thrustshift;

BOOST_AUTO_TEST_CASE(test_sort_batched_descending) {

	constexpr std::size_t N = 100;
	managed_vector<float> k_in(N);
	managed_vector<float> k_out(N);
	managed_vector<int> v_in(N);
	managed_vector<int> v_out(N);
	std::iota(v_in.begin(), v_in.end(), 0);

	random::generate_uniform_real(k_in, -10.0f, 10.0f, 1);
	std::size_t batch_len = 10;
	cudaStream_t stream = 0;
	pmr::delayed_pool_type<pmr::managed_resource_type> pool;

	async::sort_batched_descending(
	    stream, k_in, k_out, v_in, v_out, batch_len, pool);
	THRUSTSHIFT_CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
	for (std::size_t batch_id = 0; batch_id < N / batch_len; ++batch_id) {
		BOOST_TEST(std::is_sorted(k_out.rbegin() + batch_id * batch_len,
		                          k_out.rbegin() + (batch_id + 1) * batch_len));
	}
}

BOOST_AUTO_TEST_CASE(test_sort_batched_abs) {

	constexpr std::size_t N = 32;
	managed_vector<float> k_in(N);
	managed_vector<float> k_out(N);
	managed_vector<int> v_in(N);
	managed_vector<int> v_out(N);
	std::iota(v_in.begin(), v_in.end(), 0);

	random::generate_uniform_real(k_in, -10.0f, 10.0f, 1);
	std::size_t batch_len = 8;
	cudaStream_t stream = 0;
	pmr::delayed_pool_type<pmr::managed_resource_type> pool;

	async::sort_batched_abs(stream, k_in, k_out, v_in, v_out, batch_len, pool);
	THRUSTSHIFT_CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
	for (size_t batch_id = 0; batch_id < N / batch_len; ++batch_id) {

		for (size_t i = 1; i < batch_len; ++i) {
			BOOST_TEST(std::abs(k_out[(i + batch_id * batch_len) - 1]) <=
			           std::abs(k_out[i + batch_id * batch_len]));
		}
	}
}

BOOST_AUTO_TEST_CASE(test_sort_batched_abs_int) {

	constexpr std::size_t N = 32;
	managed_vector<int> k_in(N);
	managed_vector<int> k_out(N);
	managed_vector<int> v_in(N);
	managed_vector<int> v_out(N);
	std::iota(v_in.begin(), v_in.end(), 0);

	random::generate_uniform_real(k_in, -10.0f, 10.0f, 1);
	std::size_t batch_len = 8;
	cudaStream_t stream = 0;
	pmr::delayed_pool_type<pmr::managed_resource_type> pool;

	async::sort_batched_abs(stream, k_in, k_out, v_in, v_out, batch_len, pool);
	THRUSTSHIFT_CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
	for (size_t batch_id = 0; batch_id < N / batch_len; ++batch_id) {

		for (size_t i = 1; i < batch_len; ++i) {
			BOOST_TEST(std::abs(k_out[(i + batch_id * batch_len) - 1]) <=
			           std::abs(k_out[i + batch_id * batch_len]));
		}
	}
}

namespace kernel {

template <typename T>
__global__ void test_radix(const T* in, T* out, int N) {

	constexpr int num_elements_per_thread = 4;
	constexpr int block_dim = 8;
	typedef cub::BlockRadixSort<AbsView<T>, block_dim, num_elements_per_thread>
	    BlockRadixSort;
	// Allocate shared memory for BlockRadixSort
	__shared__ typename BlockRadixSort::TempStorage temp_storage;
	// Obtain a segment of consecutive items that are blocked across threads
	AbsView<T> thread_keys[num_elements_per_thread];
	const auto gtid = threadIdx.x;
	for (int i = gtid * num_elements_per_thread;
	     i < (gtid + 1) * num_elements_per_thread;
	     ++i) {
		thread_keys[i % num_elements_per_thread].value = in[i];
	}
	// Collectively sort the keys
	BlockRadixSort(temp_storage)
	    .SortDescending(thread_keys, 0, sizeof(T) * 8 - 1);
	for (int i = gtid * num_elements_per_thread;
	     i < (gtid + 1) * num_elements_per_thread;
	     ++i) {
		out[i] = thread_keys[i % num_elements_per_thread].value;
	}
}

} // namespace kernel

BOOST_AUTO_TEST_CASE(test_sort_abs_descending_kernel) {

	constexpr std::size_t N = 32;
	managed_vector<float> k_in(N);
	managed_vector<float> k_out(N);

	random::generate_uniform_real(k_in, -10.0f, 10.0f, 1);

	kernel::test_radix<float><<<1, 8>>>(k_in.data(), k_out.data(), N);
	THRUSTSHIFT_CHECK_CUDA_ERROR(cudaGetLastError());
	THRUSTSHIFT_CHECK_CUDA_ERROR(cudaDeviceSynchronize());

	for (size_t i = 1; i < N; ++i) {
		BOOST_TEST(std::abs(k_out[i - 1]) >= std::abs(k_out[i]));
	}
}
