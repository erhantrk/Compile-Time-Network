#include <iostream>

#include "Network.hpp"
int main() {
    Network<Sigmoid,1,2,3,4,5,3> net;
    net.forward({1});

    return 0;
}