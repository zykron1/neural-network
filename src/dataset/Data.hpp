#pragma once

#include <vector>
#include <random>

class Data {
	public:
		std::vector<double> x_red, y_red;
		std::vector<double> x_blue, y_blue;

		std::mt19937 gen;
		std::uniform_real_distribution<double> dist;

		Data(float lower, float upper)
        : gen(42),
          dist(lower, upper)
		{}

		void generateData();
};
