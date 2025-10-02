//
// Created by Erhan Türker on 10/2/25.
//

#pragma once
#include <algorithm>
#include <array>
#include <random>
#include <chrono>

using ActivationFn = float(*)(float);

static std::mt19937& rng() {
    thread_local std::mt19937 gen{std::random_device{}()};
    return gen;
}

template <int N>
class Neuron {
    std::array<float, N> weights{};
    float bias = 0.0f;
    float activation = 0.0f;

public:
    Neuron(float mean = 0.0f, float stddev = 1.0f) {
        std::normal_distribution dist(mean, stddev);
        for (int i = 0; i < N; ++i) weights[i] = dist(rng());
        bias = dist(rng());
    }

    Neuron(const std::array<float, N>& init_weights, float init_bias)
        : weights(init_weights), bias(init_bias) {}

    template <typename T>
    void feed(const std::array<T, N>& inputs, ActivationFn actFun) {
        float sum = bias;
        for (int i = 0; i < N; ++i) {
            sum += inputs[i].get_activation() * weights[i];
        }
        activation = actFun(sum);
    }

    void feed(const std::array<float, N>& inputs, ActivationFn actFun) {
        float sum = bias;
        for (int i = 0; i < N; ++i) {
            sum += inputs[i] * weights[i];
        }
        activation = actFun(sum);
    }



    [[nodiscard]] float get_activation() const { return activation; }
};


static float relu(float x) {
    return x > 0 ? x : 0;
}

template<int... Layers>
class Network {
    static constexpr std::array layerSizes{ Layers... };
    static constexpr int nLayers = sizeof...(Layers);


    static inline ActivationFn activationFun = relu;

    template<std::size_t I>
    using LayerType = std::array<Neuron<layerSizes[I-1]>, layerSizes[I]>;

    template<typename I>
    struct LayersTuple;

    template<std::size_t... I>
    struct LayersTuple<std::index_sequence<I...>> {
        using type = std::tuple<LayerType<I + 1>...>;
    };

    using LayersType = typename LayersTuple<std::make_index_sequence<(nLayers > 0 ? nLayers - 1 : 0)>>::type;

    LayersType net = {};

public:
    static void setActivationFunction(const ActivationFn activationFunction) {
        activationFun = activationFunction;
    }


    auto forward(const std::array<float, layerSizes[0]>& input) {
        std::for_each(std::get<0>(net).begin(), std::get<0>(net).end(),
            [&](auto& layer) {
                layer.feed(input, activationFun);
            });

        if constexpr (nLayers > 2) {
            [&]<std::size_t... Is>(std::index_sequence<Is...>) {
                ( std::for_each(std::get<Is + 1>(net).begin(), std::get<Is + 1>(net).end(),
                    [&](auto& layer) {
                        layer.feed(std::get<Is>(net), activationFun);
                    })
                , ... );
            }(std::make_index_sequence<nLayers - 2>{});
        }

        return std::get<nLayers - 2>(net);
    }

};