#pragma once

#include <exception>
#include <iostream>
#include <type_traits>
#include <utility>
#include <variant>

template <int I0, int I1, class T>
struct params_t {
	using type = T;
	static constexpr int i0 = I0;
	static constexpr int i1 = I1;
};

template <typename T, T... vars>
struct set_value {
	template <typename U>
	static void set([[maybe_unused]] U& t, [[maybe_unused]] T val) {
		std::cerr
		    << "ERROR in thrustshift/constval.h: check that your runtime value "
		       "is equal to one of the given compile time constants!"
		    << std::endl;
		std::terminate();
	}
};

template <typename T, T var, T... vars>
struct set_value<T, var, vars...> {
	template <typename U>
	static void set(U& t, T val) {
		if (var == val) {
			t = std::integral_constant<T, var>{};
		}
		else {
			set_value<T, vars...>::set(t, val);
		}
	}
};

template <typename T, T... vars>
auto make_constval(T val) {
	std::variant<std::integral_constant<T, vars>...> v;
	set_value<T, vars...>::set(v, val);
	return v;
}

template <typename T, T... vars>
struct make_type_from_value {

	template <typename... Args>
	struct get_val_from_pair {
		template <typename W>
		static void get([[maybe_unused]] W& t, [[maybe_unused]] T val) {
			std::cerr
			    << "ERROR in thrustshift/constval.h: check that your runtime "
			       "value is equal to one of the given compile time constants!"
			    << std::endl;
			std::terminate();
		}
	};

	template <typename A, typename... Args>
	struct get_val_from_pair<A, Args...> {
		template <typename W>
		static void get(W& t, T val) {
			if (A{}.first == val) {
				t = A{}.second;
			}
			else {
				get_val_from_pair<Args...>::get(t, val);
			}
		}
	};

	template <typename... U>
	static auto get(T val) {
		std::variant<U...> v;

		get_val_from_pair<
		    std::pair<std::integral_constant<T, vars>, U>...>::get(v, val);
		return v;
	}
};
