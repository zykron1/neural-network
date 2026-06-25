#include "dataset/Data.hpp"
#include "dataset/MNISTLoader.hpp"
#include "matplotlibcpp.h"
#include "dataset/Data.hpp"
#include "neural/Layer.hpp"
#include "neural/Network.hpp"
#include <iostream>

namespace plt = matplotlibcpp;

int argmax(std::vector<double>& output) {
	int bestIndex = 0;
	double bestValue = output[0];
	for (int i = 1; i < (int)output.size(); i++) {
		if (output[i] > bestValue) {
			bestValue = output[i];
			bestIndex = i;
		}
	}
	return bestIndex;
}


void show_digit_no_block(const std::vector<double>& flat, double predicted, double expected) {
    const int H = 28;
    const int W = 28;

    static std::vector<float> img(H * W);

    for (int i = 0; i < H * W; i++) {
        img[i] = static_cast<float>(flat[i]);
    }

    plt::clf();  // clear figure

    plt::imshow(
        img.data(),
        H,
        W,
        1,
        {{"cmap", "jet"}, {"interpolation", "nearest"}}
    );

    plt::title(
			"Pred: " + std::to_string((int)predicted) +
			" | Expected: " + std::to_string((int)expected),
			{{"fontsize", "32"}}
			);

    plt::axis("off");
}

int main() {
	MNISTLoader mn("test.csv");

	// Import mnist
	auto [x_train, labels] = mn.load();
	std::vector<std::vector<double>> y_train;

	// convert label to network output
	for (auto& label : labels) {
		std::vector<double> l(10);
		l[label] = 1.0;
		y_train.push_back(l);
	}

	/*
	std::vector<int> layers = {784, 64, 32, 10};
	Network n(layers);
	n.train(x_train, y_train, 0.1, 5, 32);
	n.save("mnist.model");
	*/
	
	Network n = Network::load("mnist.model");

	int success = 0;
	for (size_t i = 0; i < x_train.size(); ++i) {
		auto input = x_train[i];
		auto output = labels[i];

		auto predicted_output = n.evaluate(input);
		
		if (argmax(predicted_output) == output) {
			success++;
		}
	}

	std::cout << "Accuracy: " << success << " / " << x_train.size() << std::endl;

	plt::figure_size(300, 300);

	for (int i = 0; i < 100; i++) {
		auto input = x_train[i];
		auto output = labels[i];

		auto predicted_output = n.evaluate(input);

		show_digit_no_block(input, argmax(predicted_output), output);

		plt::pause(0.5);  // key: lets GUI update without blocking
	}
}
