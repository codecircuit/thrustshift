#include <chrono>
#include <iostream>
#include <string>
#include <tuple>
#include <vector>

#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>

#include <cuda/nvtx/profiling.hpp>
#include <cuda/runtime_api.hpp>

#include <thrustshift/k-selection.h>
#include <thrustshift/managed-vector.h>
#include <thrustshift/memory-resource.h>
#include <thrustshift/random.h>

using namespace thrustshift;
using Duration = std::chrono::duration<double, std::ratio<1, 1>>; // seconds

struct benchmark_key_t {
	int N;
	int k;
};

inline bool operator<(const benchmark_key_t& a, const benchmark_key_t& b) {
	return std::tuple{a.N, a.k} < std::tuple{b.N, b.k};
}

int main(int argc, const char* argv[]) {

	using T = float;

	auto device = cuda::device::current::get();
	auto stream = device.default_stream();

	std::cout << "  - device = \"" << device.name() << "\"" << std::endl;

	constexpr int N_max = 1 << 24;
	constexpr int k = 36;

	auto get_num_warmups = [](int N) { return 200; };
	auto get_num_measurements = [](int N) { return 1; };

	thrustshift::managed_vector<T> values_(N_max);
	gsl_lite::span<T> values(values_);
	thrustshift::random::generate_uniform_real(
	    values, -1e+7, 1e+7, 1); // last number is seed
	thrustshift::pmr::delayed_pool_type<thrustshift::pmr::managed_resource_type>
	    delayed_memory_resource;
	thrustshift::managed_vector<thrust::tuple<T, int>> selected_values_(N_max);
	gsl_lite::span<thrust::tuple<T, int>> selected_values(selected_values_);

	std::map<benchmark_key_t, double> timings;

	auto do_k_select = [&](int N, int k) {
		select_k_largest_values_abs<T>(stream,
		                               values.first(N),
		                               selected_values.first(N),
		                               k,
		                               delayed_memory_resource);
	};

	auto start = device.create_event();
	auto stop = device.create_event();

	for (int N = N_max; N >= N_max; N /= 2) {
		for (int warmup_id = 0, e = get_num_warmups(N); warmup_id < e;
		     ++warmup_id) {
			do_k_select(N, k);
		}
		cuda::profiling::start();
		start.record(stream);
		for (int measurement_id = 0, e = get_num_measurements(N);
		     measurement_id < e;
		     ++measurement_id) {
			do_k_select(N, k);
		}
		stop.record(stream);
		device.synchronize();
		cuda::profiling::stop();
		Duration dur = cuda::event::time_elapsed_between(start, stop);
		timings[{N, k}] = dur.count() / get_num_measurements(N);
	}

	std::cout << "N,k,time\n";
	for (const auto& tup : timings) {
		const auto& benchmark_key = std::get<0>(tup);
		const auto time = std::get<1>(tup);
		std::cout << benchmark_key.N << "," << benchmark_key.k << "," << time
		          << '\n';
	}
}
