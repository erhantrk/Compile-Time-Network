cmake_minimum_required(VERSION 3.31)
project(basketballsim)

set(CMAKE_CXX_STANDARD 23)

add_executable(basketballsim main.cpp
        Network.cpp
        Network.hpp
)
