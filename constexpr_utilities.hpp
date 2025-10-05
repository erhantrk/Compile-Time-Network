//
// Created by Erhan Türker on 10/5/25.
//

#pragma once

#include <utility>
template<std::size_t N, class F>
constexpr void static_for(F&& f) {
    [&]<std::size_t... I>(std::index_sequence<I...>) {
        (f.template operator()<I>(), ...);
    }(std::make_index_sequence<N>{});
}

template<std::size_t Begin, std::size_t End, class F>
constexpr void static_for(F&& f) {
    static_assert(Begin <= End);
    [&]<std::size_t... I>(std::index_sequence<I...>) {
        (f.template operator()<Begin + I>(), ...);
    }(std::make_index_sequence<End - Begin>{});
}
