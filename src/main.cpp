#include "dataset/Data.hpp"
#include "matplotlibcpp.h"
#include "dataset/Data.hpp"
#include "neural/Layer.hpp"
#include "neural/Network.hpp"

namespace plt = matplotlibcpp;

int main() {
    Data d(-10.0, 10.0);
    d.generateData();

	Network n({2,3,2});

	Layer& l0 = n.layers[0]; // 2 -> 3
	l0.weights[0*3+0] = 1.0;
	l0.weights[0*3+1] = -1.0;
	l0.weights[0*3+2] = 0.5;
	l0.weights[1*3+0] = 1.0;
	l0.weights[1*3+1] = -1.0;
	l0.weights[1*3+2] = -0.5;
	l0.biases[0] = 0.0;
	l0.biases[1] = 0.0;
	l0.biases[2] = 0.0;

	Layer& l1 = n.layers[1]; // 3 -> 2
	l1.weights[0*2+0] = 1.0;
	l1.weights[0*2+1] = -1.0;
	l1.weights[1*2+0] = -1.0;
	l1.weights[1*2+1] = 1.0;
	l1.weights[2*2+0] = 0.5;
	l1.weights[2*2+1] = -0.5;
	l1.biases[0] = 0.0;
	l1.biases[1] = 0.0;

	std::vector<double> x_red, y_red;
	std::vector<double> x_blue, y_blue;

    for (double x = -10; x <= 10; x += 0.2) {
        for (double y = -10; y <= 10; y += 0.2) {

            std::vector<double> input = {x, y};
            std::vector<double> out = n.evaluate(input);

            if (out[0] > out[1]) {
                x_red.push_back(x);
                y_red.push_back(y);
            } else {
                x_blue.push_back(x);
                y_blue.push_back(y);
            }
        }
    }
	plt::figure();

	plt::plot(x_blue, y_blue, "b.");
	plt::plot(x_red, y_red, "r.");

	plt::plot(d.x_blue, d.y_blue, "bo");
	plt::plot(d.x_red, d.y_red, "ro");

	plt::title("Neural Network");
	plt::xlabel("x");
	plt::ylabel("y");

	plt::show();

}
