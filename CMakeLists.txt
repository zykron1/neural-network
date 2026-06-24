cmake_minimum_required(VERSION 3.15)

project(NeuralNetwork LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

file(GLOB SOURCES "src/*.cpp")

add_executable(NeuralNetwork ${SOURCES})

target_include_directories(NeuralNetwork PRIVATE
    src
)

target_compile_options(NeuralNetwork PRIVATE
    -O3
)
