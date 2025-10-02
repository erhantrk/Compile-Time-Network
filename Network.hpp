//
// Created by Erhan Türker on 10/2/25.
//

#pragma once

#include <array>
#include <tuple>
#include <random>
#include <utility>
#include <algorithm>
#include <concepts>

template<class T>
concept ActivationFunction = requires(float x) {
    { T::apply(x) } -> std::same_as<float>;
};

struct ReLU {
    static float apply(float x) { return x > 0 ? x : 0; }
};

struct Sigmoid {
    static float apply(float x) { return 1.f / (1.f + std::exp(-x)); }
};

static std::mt19937& rng() {
    thread_local std::mt19937 gen{std::random_device{}()};
    return gen;
}

template<std::size_t I, int Head, int... Tail>
struct nth {
    static constexpr int value = nth<I-1, Tail...>::value;
};
template<int Head, int... Tail>
struct nth<0, Head, Tail...> {
    static constexpr int value = Head;
};

template<int In, int Out, ActivationFunction Activation>
struct Layer {
    std::array<float, Out * In> W{};
    std::array<float, Out> b{};
    std::array<float, Out> a{};

    explicit Layer(const float mean = 0.0f, const float stddev = 1.0f) {
        std::normal_distribution dist(mean, stddev);
        for (auto& w : W) w = dist(rng());
        for (auto& bi : b) bi = dist(rng());
    }

    void forward(const std::array<float, In>& x) {
        for (int j = 0; j < Out; ++j) {
            const float* row = &W[j * In];
            float sum = b[j];
            for (int i = 0; i < In; ++i) sum += row[i] * x[i];
            a[j] = Activation::apply(sum);
        }
    }
};

template<ActivationFunction Activation, int... Layers>
class Network {
    static constexpr int nLayers = sizeof...(Layers);
    static_assert(nLayers >= 2, "Need at least input and output sizes.");

    static constexpr int InputDim  = nth<0, Layers...>::value;
    static constexpr int OutputDim = nth<nLayers - 1, Layers...>::value;

    template<std::size_t I>
    using LayerType = Layer<nth<I, Layers...>::value, nth<I + 1, Layers...>::value, Activation>;

    template<typename I> struct LayersTuple;
    template<std::size_t... I>
    struct LayersTuple<std::index_sequence<I...>> {
        using type = std::tuple<LayerType<I>...>;
    };
    using LayersType = typename LayersTuple<std::make_index_sequence<nLayers-1>>::type;

    LayersType layers{};

public:
    using InputVec  = std::array<float, InputDim>;
    using OutputVec = std::array<float, OutputDim>;

    OutputVec forward(const InputVec& input) {
        std::get<0>(layers).forward(input);

        if constexpr (nLayers > 2) {
            [&]<std::size_t... I>(std::index_sequence<I...>) {
                ( (std::get<I + 1>(layers).forward(std::get<I>(layers).a)), ... );
            }(std::make_index_sequence<nLayers - 2>{});
        }

        const auto& last = std::get<nLayers - 2>(layers).a;
        return last;
    }
};
