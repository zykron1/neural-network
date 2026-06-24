#include "dataset/Data.hpp"
#include "matplotlibcpp.h"
#include "dataset/Data.hpp"
#include "neural/Layer.hpp"
#include "neural/Network.hpp"

namespace plt = matplotlibcpp;

int main() {
    Data d(-10.0, 10.0);
    d.generateData();

	std::vector<std::vector<double>> x_train;
	std::vector<std::vector<double>> y_train;

	for (size_t i = 0; i < d.x_blue.size(); i++) {
		x_train.push_back({d.x_blue[i], d.y_blue[i]});
		y_train.push_back({0.0, 1.0});
	}

	for (size_t i = 0; i < d.x_red.size(); i++) {
		x_train.push_back({d.x_red[i], d.y_red[i]});
		y_train.push_back({1.0, 0.0});
	}

	Network n({2,3,2});
	n.train(x_train, y_train, 0.01, 100);

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
