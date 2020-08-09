#pragma once

#include <string>
#include <unistd.h>

namespace thrustshift {

inline std::string hostname() {
	char name[256];
	gethostname(name, 256);
	return std::string(name);
}

}
