#pragma once

#include <random>

namespace thrustshift {

namespace random {

template <class Range, typename T = typename Range::value_type>
void generate_uniform_real(Range&& range,
                           T min = T{0},
                           T max = T{1},
                           unsigned long long seed = 0) {

	std::default_random_engine rng(seed);
	std::uniform_real_distribution<T> dist(min, max);
	for (size_t i = 0; i < range.size(); ++i) {
		range[i] = dist(rng);
	}
}

} // namespace random

} // namespace thrustshift
