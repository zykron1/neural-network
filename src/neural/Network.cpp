#include <cstddef>
#include <random>
#include <fstream>
#include <iostream>
#include <cstring>
#include <algorithm>
#include "Network.hpp"

std::vector<double> Network::evaluate(std::vector<double>& input) {
    std::vector<double> outputs = input;

    for (auto& l : layers) {
        outputs = l.calculateOutputs(outputs);
    }

    return outputs;
}

double Network::calculateError(std::vector<std::vector<double>>& inputs, std::vector<std::vector<double>>& expected_outputs) {
    double error = 0.0;
    size_t n = inputs.size();
    const double epsilon = 1e-12; 
	 
    for (size_t i = 0; i < n; i++) {
        std::vector<double> output = this->evaluate(inputs[i]);
        auto& expected_output = expected_outputs[i];
        size_t m = output.size();
        for (size_t j = 0; j < m; j++) {
            if (expected_output[j] > 0.0) {
                error -= expected_output[j] * std::log(output[j] + epsilon);
            }
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
		double stddev = std::sqrt(2.0 / static_cast<double>(layer.nodes_in));
		std::normal_distribution<double> dist(0.0, stddev);

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
	int cycles,
	size_t batchSize
) {
	this->randomizeWeights();
	size_t n = x_train.size();
	for (int i = 0; i < cycles; i++) {
		std::cout << "Cycle: " << i << std::endl;
		for (size_t batchStart = 0; batchStart < n; batchStart += batchSize) {
			size_t batchEnd = std::min(batchStart + batchSize, n);
			for (auto& layer : this->layers) {
				std::fill(layer.weightGradients.begin(), layer.weightGradients.end(), 0.0);
				std::fill(layer.biasGradients.begin(), layer.biasGradients.end(), 0.0);
			}
			for (size_t s = batchStart; s < batchEnd; s++) {
				this->evaluate(x_train[s]);
				int lastIndex = (int)this->layers.size() - 1;
				std::vector<double> nodeValues = this->layers[lastIndex].calculateOutputNodeValues(y_train[s]);
				this->layers[lastIndex].updateGradients();
				for (int l = lastIndex - 1; l >= 0; l--) {
					nodeValues = this->layers[l].calculateHiddenNodeValues(this->layers[l + 1], nodeValues);
					this->layers[l].updateGradients();
				}
			}
			this->applyAllGradients(learn_rate / (double)(batchEnd - batchStart));
		}
	}
}

void Network::save(const std::string& path) {
	std::ofstream file(path, std::ios::binary);

	if (!file)
		throw std::runtime_error("Failed to open file");

	// Magic number
	const char magic[4] = {'N', 'N', 'E', 'T'};
	file.write(magic, 4);

	uint32_t version = 1;
	file.write(reinterpret_cast<char*>(&version), sizeof(version));

	size_t layerCount = layers.size();
	file.write(reinterpret_cast<char*>(&layerCount), sizeof(layerCount));

	for (const Layer& layer : layers) {
		file.write(reinterpret_cast<const char*>(&layer.nodes_in),
				sizeof(layer.nodes_in));

		file.write(reinterpret_cast<const char*>(&layer.nodes_out),
				sizeof(layer.nodes_out));

		file.write(reinterpret_cast<const char*>(layer.weights.data()),
				layer.weights.size() * sizeof(double));

		file.write(reinterpret_cast<const char*>(layer.biases.data()),
				layer.biases.size() * sizeof(double));
	}
}

Network Network::load(const std::string& path) {
    std::ifstream file(path, std::ios::binary);

    if (!file)
        throw std::runtime_error("Failed to open file");

    char magic[4];
    file.read(magic, 4);

    if (memcmp(magic, "NNET", 4) != 0)
        throw std::runtime_error("Invalid model file");

    uint32_t version;
    file.read(reinterpret_cast<char*>(&version), sizeof(version));

    if (version != 1)
        throw std::runtime_error("Unsupported version");

    size_t layerCount;
    file.read(reinterpret_cast<char*>(&layerCount), sizeof(layerCount));

    Network net;

    for (size_t i = 0; i < layerCount; i++) {
        Layer layer;

		file.read(reinterpret_cast<char*>(&layer.nodes_in), sizeof(layer.nodes_in));
		file.read(reinterpret_cast<char*>(&layer.nodes_out), sizeof(layer.nodes_out));
		layer.weights.resize(layer.nodes_in * layer.nodes_out);
        layer.biases.resize(layer.nodes_out);

        file.read(reinterpret_cast<char*>(layer.weights.data()),
			layer.weights.size() * sizeof(double));

        file.read(reinterpret_cast<char*>(layer.biases.data()),
            layer.biases.size() * sizeof(double));

		if (i + 1 == layerCount) layer.isOutputLayer = true;
        net.layers.push_back(std::move(layer));
    }

    return net;
}
