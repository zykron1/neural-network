cmake_minimum_required(VERSION 3.15)

project(NeuralNetwork LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python3 COMPONENTS Development NumPy REQUIRED)

file(GLOB_RECURSE SOURCES "src/*.cpp")

add_executable(NeuralNetwork ${SOURCES})

target_include_directories(NeuralNetwork PRIVATE
    ${Python3_INCLUDE_DIRS}
    ${Python3_NumPy_INCLUDE_DIRS}
    src
)

target_compile_options(NeuralNetwork PRIVATE
    -O3
)

target_link_libraries(NeuralNetwork PRIVATE
    Python3::Python
)
