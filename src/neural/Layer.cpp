#include "Layer.hpp"
#include <vector>

std::vector<double> Layer::calculateOutputs(std::vector<double>& inputs) {
	std::vector<double> output(this->nodes_out);

	for (int nodeOut = 0; nodeOut < this->nodes_out; nodeOut++) {
		double weighted = this->biases[nodeOut];

		for (int nodeIn = 0; nodeIn < this->nodes_in; nodeIn++) {
			weighted += inputs[nodeIn] * this->weight(nodeIn, nodeOut);
		}

		output[nodeOut] = weighted;
	}

	return output;
}

double& Layer::weight(std::size_t in_node, std::size_t out_node) {
	return weights[in_node * nodes_out + out_node];
}

double& Layer::weightGradient(std::size_t in_node, std::size_t out_node) {
	return this->weightGradients[in_node * nodes_out + out_node];
}

void Layer::applyGradients(double learnRate) {
	for (int nodeOut = 0; nodeOut < this->nodes_out; nodeOut++) {
		biases[nodeOut] -= this->biasGradients[nodeOut] * learnRate;
		for (int nodeIn = 0; nodeIn < this->nodes_in; nodeIn++) {
			this->weight(nodeIn, nodeOut) -= this->weightGradient(nodeIn, nodeOut) * learnRate;
		}

	}
}
