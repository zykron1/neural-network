#pragma once

#include <string>
#include "Layer.hpp"

class Network {
	private:
		double calculateError(std::vector<std::vector<double>>& inputs, std::vector<std::vector<double>>& expected_outputs);
		void applyAllGradients(double learnRate);
		void randomizeWeights();

	public:
		std::vector<Layer> layers;
		Network(const std::vector<int>& layerSizes) {
			this->layers.reserve(size(layerSizes)-1);
			for (int i = 0; i < size(layerSizes) - 1; ++i) {
				this->layers.emplace_back(layerSizes[i], layerSizes[i+1]);
			}
			this->layers.back().isOutputLayer = true;
		}
		Network() = default;

		std::vector<double> evaluate(std::vector<double>& inputs);
		
		void train(
			std::vector<std::vector<double>>& x_train,
			std::vector<std::vector<double>>& y_train,
			double learn_rate,
			int cycles,
			std::size_t batchSize
		);

		static Network load(const std::string& path);
		void save(const std::string& path);
};
