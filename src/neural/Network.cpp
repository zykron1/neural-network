#include "Network.hpp"
#include <cstddef>
#include <random>
#include <iostream>

std::vector<double> Network::evaluate(std::vector<double>& input) {
    std::vector<double> outputs = input;

    for (auto& l : layers) {
        outputs = l.calculateOutputs(outputs);
    }

    return outputs;
}

std::vector<std::vector<double>> Network::evaluateInputs(std::vector<std::vector<double>>& inputs) {
    std::vector<std::vector<double>> outputs;

	for (auto input : inputs) {
		outputs.push_back(this->evaluate(input));
	}

    return outputs;
}

double Network::calculateError(std::vector<std::vector<double>>& inputs, std::vector<std::vector<double>>& expected_outputs) {
	auto outputs = this->evaluateInputs(inputs);

	double error = 0.0;
	size_t n = outputs.size();

	for (int i = 0; i < n; i++) {
		auto output = outputs[i];
		auto expected_output = expected_outputs[i];

		size_t m = output.size();

		for (int j = 0; j < m; j++) {
            double diff = output[j] - expected_output[j];
            error += diff * diff;
        }
	}


	return error / n;
}

void Network::applyAllGradients(double learnRate) {
	for (auto& l : this->layers) {
		l.applyGradients(learnRate);
	}
}

void Network::randomizeWeights() {
	std::random_device rd;
	std::mt19937 gen(rd());

	for (Layer& layer : layers) {
		// Xavier/Glorot-style range based on number of inputs to the layer
		double limit = 1.0 / std::sqrt(static_cast<double>(layer.nodes_in));
		std::uniform_real_distribution<double> dist(-limit, limit);

		for (std::size_t i = 0; i < layer.nodes_in; ++i) {
			for (std::size_t o = 0; o < layer.nodes_out; ++o) {
				layer.weight(i, o) = dist(gen);
			}
		}
	}
}

void Network::train(
	std::vector<std::vector<double>>& x_train,
	std::vector<std::vector<double>>& y_train,
	double learn_rate,
	int cycles
) {

	this->randomizeWeights();
	for (int i = 0; i < cycles; i++) {
		double originalError = this->calculateError(x_train, y_train);
		const double h = 0.0001; 

		for (auto& layer : this->layers) {
			for (int nodeOut = 0; nodeOut < layer.nodes_out; nodeOut++) {
				for (int nodeIn = 0; nodeIn < layer.nodes_in; nodeIn++) {
					layer.weight(nodeIn, nodeOut) += h; // change node

					// f(x+h)-f(x) / h lim h-> 0
					double deltaError = this->calculateError(x_train, y_train) - originalError;
					layer.weightGradient(nodeIn, nodeOut) = deltaError / h; 

					layer.weight(nodeIn, nodeOut) -= h; // undo change to not affect other nodes
				}
			}

			for (int b = 0; b < size(layer.biases); b++) {
				layer.biases[b] += h;

				double deltaError = this->calculateError(x_train, y_train) - originalError;
				layer.biasGradients[b] = deltaError / h;

				layer.biases[b] -= h;
			}


		}
		this->applyAllGradients(learn_rate);
	}
}
