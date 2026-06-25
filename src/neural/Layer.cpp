#include "Layer.hpp"
#include <vector>

std::vector<double> Layer::calculateOutputs(std::vector<double>& inputs) {
	this->lastInputs = inputs;
	std::vector<double> weightedInputs(this->nodes_out);
	std::vector<double> output(this->nodes_out);
	for (int nodeOut = 0; nodeOut < this->nodes_out; nodeOut++) {
		double weighted = this->biases[nodeOut];
		for (int nodeIn = 0; nodeIn < this->nodes_in; nodeIn++) {
			weighted += inputs[nodeIn] * this->weight(nodeIn, nodeOut);
		}
		weightedInputs[nodeOut] = weighted;
	}
	this->lastWeightedInputs = weightedInputs;
	if (this->isOutputLayer) {
		double maxVal = weightedInputs[0];
		for (int nodeOut = 1; nodeOut < this->nodes_out; nodeOut++) {
			if (weightedInputs[nodeOut] > maxVal) maxVal = weightedInputs[nodeOut];
		}
		double sum = 0.0;
		for (int nodeOut = 0; nodeOut < this->nodes_out; nodeOut++) {
			output[nodeOut] = std::exp(weightedInputs[nodeOut] - maxVal);
			sum += output[nodeOut];
		}
		for (int nodeOut = 0; nodeOut < this->nodes_out; nodeOut++) {
			output[nodeOut] /= sum;
		}
	} else {
		for (int nodeOut = 0; nodeOut < this->nodes_out; nodeOut++) {
			output[nodeOut] = relu(weightedInputs[nodeOut]);
		}
	}
	this->lastOutputs = output;
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

std::vector<double> Layer::calculateOutputNodeValues(std::vector<double>& expected_outputs) {
	std::vector<double> nodeValues(this->nodes_out);
	for (int nodeOut = 0; nodeOut < this->nodes_out; nodeOut++) {
		nodeValues[nodeOut] = this->lastOutputs[nodeOut] - expected_outputs[nodeOut];
	}
	this->nodeValues = nodeValues;
	return nodeValues;
}

std::vector<double> Layer::calculateHiddenNodeValues(Layer& nextLayer, std::vector<double>& nextNodeValues) {
	std::vector<double> nodeValues(this->nodes_out);
	for (int nodeOut = 0; nodeOut < this->nodes_out; nodeOut++) {
		double sum = 0.0;
		for (int nextNodeOut = 0; nextNodeOut < nextLayer.nodes_out; nextNodeOut++) {
			sum += nextLayer.weight(nodeOut, nextNodeOut) * nextNodeValues[nextNodeOut];
		}
		nodeValues[nodeOut] = sum * step(this->lastWeightedInputs[nodeOut]);
	}
	this->nodeValues = nodeValues;
	return nodeValues;
}

void Layer::updateGradients() {
	for (int nodeOut = 0; nodeOut < this->nodes_out; nodeOut++) {
		this->biasGradients[nodeOut] += this->nodeValues[nodeOut];
		for (int nodeIn = 0; nodeIn < this->nodes_in; nodeIn++) {
			this->weightGradient(nodeIn, nodeOut) += this->lastInputs[nodeIn] * this->nodeValues[nodeOut];
		}
	}
}
