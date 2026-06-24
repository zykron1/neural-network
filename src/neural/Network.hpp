#pragma once

#include "Layer.hpp"

class Network {
	public:
		std::vector<Layer> layers;

		Network(const std::vector<int>& layerSizes) {
			this->layers.reserve(size(layerSizes)-1);

			for (int i = 0; i < size(layerSizes) - 1; ++i) {
				this->layers.emplace_back(layerSizes[i], layerSizes[i+1]);
			}
		}

		std::vector<double> evaluate(std::vector<double> inputs);
};
