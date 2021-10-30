#pragma once

namespace thrustshift {

//! nvcc defines `warpSize`, but it is not a compile time constant
constexpr int warp_size = 32;

} // namespace thrustshift
