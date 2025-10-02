#include <iostream>

#include "Network.hpp"
int main() {
    Network<1,2,2,2,2> net;
    net.forward({1});

    return 0;
}