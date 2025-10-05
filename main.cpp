#include <chrono>
#include <iostream>

#include "Network.hpp"

int main() {
    using Net = Network<ReLU, 12,128,128,8>;
    Net net;

    constexpr int N = 10'000'000;
    const auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < N; ++i) {
        Net::InputVec input = {1,2,3,4};
        const auto out = net.forward(input);
        (void)out; // prevent optimizing away
    }

    const auto end = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double> elapsed = end - start;

    std::cout << "Ran " << N << " forward passes in "
              << elapsed.count() << " seconds\n";
    std::cout << "Average per forward: "
              << (elapsed.count() / N) * 1e9 << " ns\n";

    return 0;
}