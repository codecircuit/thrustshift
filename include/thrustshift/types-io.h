#pragma once

#include <string>

#include <thrustshift/for-each.h>

namespace thrustshift {

namespace tuple {

template <class Tuple>
std::string to_string(Tuple&& tup) {
	std::stringstream ss;
	auto f = [&](const auto& arg) { ss << arg << ','; };
	for_each(tup, f);
	auto str = ss.str();
	str.pop_back(); // erase last comma
	return str;
}

} // namespace tuple

} // namespace thrustshift
