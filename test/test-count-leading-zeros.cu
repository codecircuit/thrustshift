#include <algorithm>
#include <variant>
#include <vector>

#include <Eigen/Sparse>

#include <gsl-lite/gsl-lite.hpp>

#include <cuda/runtime_api.hpp>

#include <boost/test/data/test_case.hpp>
#include <boost/test/unit_test.hpp>

#include <thrustshift/CSR.h>
#include <thrustshift/bit.h>
#include <thrustshift/managed-vector.h>
#include <thrustshift/memory-resource.h>

namespace bdata = boost::unit_test::data;
using namespace thrustshift;

namespace kernel {
template <typename I>
__global__ void clz(I i, int* result) {

	*result = count_leading_zeros(i);
}

} // namespace kernel

BOOST_AUTO_TEST_CASE(test_count_leading_zeros) {

	auto device = cuda::device::current::get();
	auto c = cuda::make_launch_config(1, 1);
	auto device_result = cuda::memory::managed::make_unique<int>();

	{
		const int i = 0b00000000000000000000000000000000;
		const int gold_result = 32;
		cuda::launch(kernel::clz<int>, c, i, device_result.get());
		device.synchronize();
		BOOST_TEST(*device_result == gold_result);
	}
	{
		const int i = 0b00000000000000000000000000000011;
		const int gold_result = 30;
		cuda::launch(kernel::clz<int>, c, i, device_result.get());
		device.synchronize();
		BOOST_TEST(*device_result == gold_result);
	}
	{
		const long long int i = 0b0000000000000000000000000000001100000000000000000000000000000011;
		const long long int gold_result = 30;
		cuda::launch(kernel::clz<long long int>, c, i, device_result.get());
		device.synchronize();
		BOOST_TEST(*device_result == gold_result);
	}

	{
		const long long int i = 0b0000000000000000000000000000000000000000000000000000000000000000;
		const long long int gold_result = 64;
		cuda::launch(kernel::clz<long long int>, c, i, device_result.get());
		device.synchronize();
		BOOST_TEST(*device_result == gold_result);
	}

}
