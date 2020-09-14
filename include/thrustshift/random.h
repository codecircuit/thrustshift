#pragma once

#include <random>
#include <type_traits>

#include <gsl-lite/gsl-lite.hpp>

namespace thrustshift {

namespace random {

template <class Range, class Distribution, class Engine>
void generate(Range&& range, Distribution& dist, Engine& rng) {
	for (size_t i = 0; i < range.size(); ++i) {
		range[i] = dist(rng);
	}
}

template <class Range,
          typename T = typename std::remove_reference<Range>::type::value_type>
void generate_uniform_real(Range&& range,
                           T min = T{0},
                           T max = T{1},
                           unsigned long long seed = 0) {
	gsl_Expects(min <= max);
	std::default_random_engine rng(seed);
	std::uniform_real_distribution<T> dist(min, max);
	generate(std::forward<Range>(range), dist, rng);
}

template <class Range,
          typename T = typename std::remove_reference<Range>::type::value_type>
void generate_normal_real(Range&& range,
                          T mean = T{0},
                          T stddev = T{1},
                          unsigned long long seed = 0) {
	std::default_random_engine rng(seed);
	std::normal_distribution<T> dist(mean, stddev);
	generate(std::forward<Range>(range), dist, rng);
}

} // namespace random

} // namespace thrustshift
