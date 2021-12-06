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
#include <thrustshift/prefetch.h>
#include <thrustshift/random.h>

using namespace thrustshift;
using Duration = std::chrono::duration<double, std::ratio<1, 1>>; // seconds

struct benchmark_key_t {
	int N;
	int k;
	int scheme_id;
};

inline bool operator<(const benchmark_key_t& a, const benchmark_key_t& b) {
	return std::tuple{a.N, a.k, a.scheme_id} <
	       std::tuple{b.N, b.k, b.scheme_id};
}

template <typename T>
auto select_k_values_cpu(gsl_lite::span<const T> values, int k) {
	std::set<std::tuple<T, T, int>> s;
	T curr_min = 0;
	for (int i = 0; i < k; ++i) {
		const auto xs = values[i];
		const auto x = std::abs(xs);
		s.insert({x, xs, i});
		curr_min = std::min(curr_min, x);
	}
	for (int i = k, e = values.size(); i < e; ++i) {
		const auto xs = values[i];
		const auto x = std::abs(xs);
		if (x > curr_min) {
			s.erase(s.begin());
			s.insert({x, xs, i});
			curr_min = std::get<0>(*s.begin());
		}
	}
	return s;
}

int main(int argc, const char* argv[]) {

	using T = float;

	auto device = cuda::device::current::get();
	auto stream = device.default_stream();

	std::cout << "  - device = \"" << device.name() << "\"" << std::endl;

	constexpr int N_max = 1 << 20;
	constexpr int N_min = 1 << 20;
	constexpr int k = 36;

	auto get_num_warmups = [&](int N) { return std::stoi(argv[1]); };
	auto get_num_measurements = [&](int N) { return std::stoi(argv[2]); };

	std::cout << "  - num_warmups = " << get_num_warmups(100) << std::endl;
	std::cout << "  - num_measurements = " << get_num_measurements(100)
	          << std::endl;

	thrustshift::managed_vector<T> values_(N_max);
	gsl_lite::span<T> values(values_);
	thrustshift::random::generate_uniform_real(
	    values, -1e+7, 1e+7, 1); // last number is seed
	thrustshift::pmr::delayed_pool_type<thrustshift::pmr::managed_resource_type>
	    delayed_memory_resource;
	thrustshift::managed_vector<T> selected_values_(N_max);
	thrustshift::managed_vector<int> selected_indices_(N_max);

	thrustshift::managed_vector<int> bit_offset_s(1);
	thrustshift::managed_vector<uint64_t> prefix_s(1);
	thrustshift::async::prefetch(stream, device, bit_offset_s);
	thrustshift::async::prefetch(stream, device, prefix_s);

	gsl_lite::span<T> selected_values(selected_values_);
	gsl_lite::span<int> selected_indices(selected_indices_);

	std::map<benchmark_key_t, double> timings;
	std::tuple<uint64_t, int> tup;
	std::set<std::tuple<T, T, int>> set;
	auto do_k_select = [&](int N, int k, int scheme) {
		if (scheme == 0) {
			select_k_largest_values_abs<T>(stream,
			                               values.first(N),
			                               selected_values.first(N),
			                               selected_indices.first(N),
			                               k,
			                               delayed_memory_resource);
		}
		else if (scheme == 1) {
			cooperative::select_k_largest_values_abs<T>(
			    stream,
			    values.first(N),
			    selected_values.first(N),
			    selected_indices.first(N),
			    k,
			    delayed_memory_resource);
		}
		else if (scheme == 2) {
			cooperative::select_k_largest_values_abs2<T>(
			    stream,
			    values.first(N),
			    selected_values.first(N),
			    selected_indices.first(N),
			    k,
			    delayed_memory_resource);
		}
		else if (scheme == 3) {
			select_k_largest_values_abs_with_cub<T>(stream,
			                                        values.first(N),
			                                        selected_values.first(N),
			                                        selected_indices.first(N),
			                                        k,
			                                        delayed_memory_resource);
		}
		else if (scheme == 4) {
			dynamic_parallelism::select_k_largest_values_abs<T>(
			    stream,
			    values.first(N),
			    selected_values.first(N),
			    selected_indices.first(N),
			    k,
			    delayed_memory_resource);
		}
		else if (scheme == 5) {
			tup = k_largest_values_abs_radix_atomic<T>(
			    stream, values.first(N), k, delayed_memory_resource);
		}
		else if (scheme == 6) {
			tup = k_largest_values_abs_radix_atomic_devicehisto<T>(
			    stream, values.first(N), k, delayed_memory_resource);
		}
		else if (scheme == 7) {
			tup =
			    dynamic_parallelism::k_largest_values_abs_radix_atomic_binning<
			        T>(stream,
			           values.first(N),
			           k,
			           false, // with sync
			           delayed_memory_resource);
		}
		else if (scheme == 8) {
			tup =
			    dynamic_parallelism::k_largest_values_abs_radix_atomic_binning<
			        T>(stream,
			           values.first(N),
			           k,
			           true, // without sync
			           delayed_memory_resource);
		}
		else if (scheme == 9) {
			tup = k_largest_values_abs_radix<T>(
			    stream, values.first(N), k, delayed_memory_resource);
		}
		else if (scheme == 10) {
			async::k_largest_values_abs_radix_atomic_devicehisto_with_ptr<T>(
			    stream,
			    values.first(N),
			    prefix_s.data(),
			    bit_offset_s.data(),
			    k,
			    delayed_memory_resource);
		}
		else if (scheme == 11) {
			async::select_k_largest_values_abs<T>(stream,
			                                      values.first(N),
			                                      selected_values.first(N),
			                                      selected_indices.first(N),
			                                      k,
			                                      delayed_memory_resource);
		}
		else if (scheme == 12) {
			set = select_k_values_cpu<T>(values, k);
		}
		else {
			std::terminate();
		}
	};

	auto start = device.create_event();
	auto stop = device.create_event();

	for (int N = N_max; N >= N_min; N /= 2) {
		for (int scheme_id : std::array{11}) {
			for (int warmup_id = 0, e = get_num_warmups(N); warmup_id < e;
			     ++warmup_id) {
				do_k_select(N, k, scheme_id);
			}
			cuda::profiling::start();
			do_k_select(N, k, scheme_id);
			cuda::profiling::stop();
			const std::string range_label(std::to_string(scheme_id));
			auto range = cuda::profiling::mark::range_start(range_label);
			start.record(stream);
			for (int measurement_id = 0, e = get_num_measurements(N);
			     measurement_id < e;
			     ++measurement_id) {
				do_k_select(N, k, scheme_id);
			}
			stop.record(stream);
			device.synchronize();
			cuda::profiling::mark::range_end(range);
			// std::cout << "tup = " << std::get<0>(tup) << std::endl;
			Duration dur = cuda::event::time_elapsed_between(start, stop);
			timings[{N, k, scheme_id}] = dur.count() / get_num_measurements(N);
		}
	}

	std::cout << "set size = " << set.size() << std::endl;
	std::cout << "N,k,scheme_id,time\n";
	for (const auto& tup : timings) {
		const auto& benchmark_key = std::get<0>(tup);
		const auto time = std::get<1>(tup);
		std::cout << benchmark_key.N << "," << benchmark_key.k << ","
		          << benchmark_key.scheme_id << "," << time << '\n';
	}
}
