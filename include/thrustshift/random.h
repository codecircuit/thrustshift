#pragma once

#include <random>
#include <type_traits>

namespace thrustshift {

namespace random {

template <class Range,
	  typename T = typename std::remove_reference<Range>::type::value_type>
void generate_uniform_real(Range&& range, T min = T{0}, T max = T{1},
			   unsigned long long seed = 0) {
    std::default_random_engine rng(seed);
    std::uniform_real_distribution<T> dist(min, max);
    for (size_t i = 0; i < range.size(); ++i) {
	range[i] = dist(rng);
    }
}

}  // namespace random

}  // namespace thrustshift
