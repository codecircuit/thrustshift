# Thrustshift library

CUDA library about what I consider useful and generic functions.

## Dependencies

* [CMakeshift](https://github.com/mbeutel/CMakeshift)
* [gsl-lite](https://github.com/gsl-lite/gsl-lite), an implementation of the [C++ Core Guidelines Support Library](https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines#S-gsl)
* [Eigen3](https://gitlab.com/libeigen/eigen)

## Design Concept

### Ranges instead of iterators

A common design of the STL and thrust is to deploy iterators to access data. E.g. the function `copy`

```cpp
copy(src_begin, src_end, dst_begin);
```

The function cannot check if the range accessed by `dst_begin` is of the same
length as the source range, nor can be checked that `src_begin` and `src_end` point
to the same range. Thrustshift makes use of the concept of a range, which provides
a length and addresses the latter mentioned drawbacks.

```cpp
thrustshift::async::copy(stream, src, dst);
```

### Asynchronous namespace

Where possible Thrustshift provides asynchronous functions in the `async` namespace.
The first argument to these functions is always a `cudaStream_t`.
E.g. Thrust only provides synchronous gather and scatter functions, which synchronize with
the device.

### Polymorphic memory resources

Thrustshift adapts the concept of the [polymorphic memory resources](https://en.cppreference.com/w/cpp/memory/memory_resource) of the STL
to provide a full configurable interface regarding the memory usage of functions. Some functions might
need temporary memory, which can be allocated with the given memory resource. E.g.

```cpp
thrustshift::async::reduce(stream, values, result, reduction_functor, initial_value, delayed_memory_resource);
```

The `delayed_memory_resource` is only allowed to deallocate the memory, which was allocated by `thrustshift::async::reduce` **after**
all calls to `stream`, which were queued by `thrustshift::async::reduce`, are finished. Although the `deallocate` is already called
for all buffers, which were allocated by `thrustshift::async::reduce`. If you just want running code use:

```cpp
#include <thrustshift/memory_resource.h>
#include <thrustshift/reduction.h>

...
thrustshift::pmr::delayed_pool_type<thrustshift::pmr::managed_resource_type> delayed_memory_resource;
...
thrustshift::async::reduce(stream, values, result, reduction_functor, initial_value, delayed_memory_resource);
...

```

#### Other design concepts regarding temporarily required buffers

[cuSPARSE](https://docs.nvidia.com/cuda/cusparse/index.html) and [CUB](https://nvlabs.github.io/cub/) either
use a different function call to determine the size of the temporary buffer or just return the size of the
temporary buffer on the first function call. Both designs make the nesting of functions, which require
temporary buffers inherently difficult. E.g. if you write a function, which uses two other functions, which
require temporary buffers, but the new function should only expose one `void* tmp_buffer` you must take
care about alignment if the two other functions require different types of temporary buffers. The emerging code
is difficult to read and blows up the code size.

## How to build the project

1. Clone the Dependencies to your machine and build the projects

2. Clone the repository to your preferred location

3. Create a build folder

```bash
take build/release # zsh shell
```
4. Configure with CMake

In the following the library is built for CUDA architecture `sm_75`.
Please adjust accordingly for your GPU architecture.

```bash
cmake ... -DCMAKE_BUILD_TYPE=Release
  -DCMakeshift_DIR=$PATH_TO_CMAKESHIFT_BUILD_DIR
  -Dgsl-lite_DIR=$PATH_TO_GSL_LITE_BUILD_DIR
  -DCMAKE_CUDA_ARCHITECTURES="75"
```

Alternatively to declaring the paths to the dependencies explicitly, you can install them and
set the `CMAKE_PREFIX_PATH` accordingly.

6. Make

```bash
make -j
```

Thrustshift is a header-only library. Therefore, 'building' only creates the
CMake config files.
