#include <cstring>

#include <array>
#include <chrono>
#include <iostream>
#include <string>
#include <tuple>
#include <vector>

#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>

#include <cuda/nvtx/profiling.hpp>
#include <cuda/runtime_api.hpp>

#include <makeshift/iomanip.hpp>

#include <thrustshift/k-selection.h>
#include <thrustshift/managed-vector.h>
#include <thrustshift/memory-resource.h>
#include <thrustshift/random.h>

namespace po = boost::program_options;
using namespace thrustshift;
using Duration = std::chrono::duration<double, std::ratio<1, 1>>; // seconds

enum class scheme_t { vanilla, threadfence, atomic, cub };

constexpr auto reflect(gsl_lite::type_identity<scheme_t>) {
	return std::array{
	    std::pair{scheme_t::vanilla, "vanilla"},
	    std::pair{scheme_t::threadfence, "threadfence"},
	    std::pair{scheme_t::atomic, "atomic"},
	    std::pair{scheme_t::cub, "cub"},
	};
}

inline std::ostream& operator<<(std::ostream& stream, scheme_t c) {
	return stream << makeshift::as_enum(c);
}
inline std::istream& operator>>(std::istream& stream, scheme_t& c) {
	return stream >> makeshift::as_enum(c);
}

struct cli_params_t {
	int N = 1 << 20;
	int k = 36;
	int num_histograms = 68 * 4;
	int num_sh_histograms = 1;
	int block_dim = 256;
	int num_measurements = 500;
	int num_warmups = 50;
	scheme_t scheme = scheme_t::vanilla;
};

po::variables_map parse_cmdline(const int argc,
                                const char* argv[],
                                cli_params_t& clip) {
	po::options_description desc("Usage: " + std::string(argv[0]) +
	                             "\nAllowed Options:");
	desc.add_options()("help", "show help message");

	desc.add_options()("N", po::value<int>(&clip.N), "number of values");
	desc.add_options()(
	    "k", po::value<int>(&clip.k), "number of selected values");
	desc.add_options()("num_histograms",
	                   po::value<int>(&clip.num_histograms),
	                   "grid dimension of bin_values256; each block fills one "
	                   "single histogram");
	desc.add_options()("num_sh_histograms",
	                   po::value<int>(&clip.num_sh_histograms),
	                   "number of histograms in shared memory");
	desc.add_options()("block_dim",
	                   po::value<int>(&clip.block_dim),
	                   "block dim of bin_values256 kernel");
	desc.add_options()("scheme", po::value<scheme_t>(&clip.scheme), "");

	po::variables_map vm;
	po::store(po::parse_command_line(argc, argv, desc), vm);
	po::notify(vm);

	if (vm.count("help")) {
		std::cout << desc << std::endl;
		exit(EXIT_SUCCESS);
	}

	return vm;
}

int main(int argc, const char* argv[]) {

	using T = float;

	cli_params_t clip;
	auto vm = parse_cmdline(argc, argv, clip);

	auto device = cuda::device::current::get();
	auto stream = device.default_stream();

	std::cout << "  - device = \"" << device.name() << "\"" << std::endl;
	std::cout << "  - N = " << clip.N << std::endl;

	thrustshift::managed_vector<T> values_(clip.N);
	gsl_lite::span<T> values(values_);
	thrustshift::random::generate_uniform_real(
	    values, -1e+7, 1e+7, 1); // last number is seed
	thrustshift::pmr::delayed_pool_type<thrustshift::pmr::managed_resource_type>
	    delayed_memory_resource;
	thrustshift::managed_vector<int> histogram(256);

	auto unary_functor = [] __host__ __device__(T x) { return abs(x); };
	auto bin_index_transform = [] __host__ __device__(int i) { return i; };

	auto exec = [&] {
		constexpr std::array<uint64_t, 4> prefixes{0, 75, 19224, 4921493};
		for (int bit_offset = 0, i = 0; bit_offset < 32; bit_offset += 8, ++i) {
			if (clip.scheme == scheme_t::vanilla) {
				async::bin_values256<T>(stream,
				                        values,
				                        histogram,
				                        bit_offset,
				                        prefixes[i],
				                        clip.block_dim,
				                        clip.num_histograms,
				                        clip.num_sh_histograms,
				                        unary_functor,
				                        bin_index_transform,
				                        delayed_memory_resource);
			}
			else if (clip.scheme == scheme_t::threadfence) {
				async::bin_values256_threadfence<T>(stream,
				                                    values,
				                                    histogram,
				                                    bit_offset,
				                                    prefixes[i],
				                                    clip.block_dim,
				                                    clip.num_histograms,
				                                    clip.num_sh_histograms,
				                                    unary_functor,
				                                    bin_index_transform,
				                                    delayed_memory_resource);
			}
			else if (clip.scheme == scheme_t::atomic) {
				async::bin_values256_atomic<T>(stream,
				                               values,
				                               histogram,
				                               bit_offset,
				                               prefixes[i],
				                               clip.block_dim,
				                               clip.num_histograms,
				                               clip.num_sh_histograms,
				                               unary_functor,
				                               bin_index_transform,
				                               delayed_memory_resource);
			}
			else if (clip.scheme == scheme_t::cub) {
				constexpr int histogram_length = 256;
				const uint64_t prefix = prefixes[i];
				using I =
				    typename thrustshift::make_uintegral_of_equal_size<T>::type;
				I lower_level = 0; // inclusive
				I upper_level = std::numeric_limits<I>::max(); // exclusive
				const int N = values.size();
				if (bit_offset > 0) {
					lower_level = prefix << (sizeof(I) * 8 - (bit_offset + 8));
					upper_level = (prefix + 1)
					              << (sizeof(I) * 8 - (bit_offset + 8));
				}

				auto sample_iterator = thrust::make_transform_iterator(
				    values.data(),
				    [prefix, bit_offset] __host__ __device__(const T& x) {
					    using std::abs;
					    const T abs_x = abs(x);
					    I k;
					    std::memcpy(&k, &abs_x, sizeof(I));
					    return k;
				    });
				async::bin_values_into_histogram(stream,
				                                 sample_iterator,
				                                 histogram.data(),
				                                 histogram_length,
				                                 lower_level,
				                                 upper_level,
				                                 N,
				                                 delayed_memory_resource);
			}
			else {
				std::terminate();
			}
		}
	};

	auto start = device.create_event();
	auto stop = device.create_event();

	for (int warmup_id = 0, e = clip.num_warmups; warmup_id < e; ++warmup_id) {
		exec();
	}

	cuda::profiling::start();
	exec();
	cuda::profiling::stop();
	start.record(stream);
	for (int measurement_id = 0, e = clip.num_measurements; measurement_id < e;
	     ++measurement_id) {
		exec();
	}
	stop.record(stream);
	device.synchronize();
	Duration dur = cuda::event::time_elapsed_between(start, stop);
	std::cout << "  - duration = " << dur.count() / clip.num_measurements
	          << " s" << std::endl;
}
