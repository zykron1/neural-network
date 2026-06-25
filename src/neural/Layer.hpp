#pragma once

#include <vector>
#include <math.h>

class Layer {
	public:
		std::size_t nodes_in, nodes_out;
		std::vector<double> weights;
		std::vector<double> weightGradients;
		std::vector<double> biases;
		std::vector<double> biasGradients;

		bool isOutputLayer = false;
		std::vector<double> lastInputs;
		std::vector<double> lastWeightedInputs;
		std::vector<double> lastOutputs;
		std::vector<double> nodeValues;
		
		Layer(std::size_t in, std::size_t out) :
			nodes_in(in),
			nodes_out(out),
			weights(in*out),
			weightGradients(in*out),
			biases(out),
			biasGradients(out)
		{} 
		Layer() = default;

		double& weight(std::size_t in_node, std::size_t out_node);
		double& weightGradient(std::size_t in_node, std::size_t out_node);

		std::vector<double> calculateOutputs(std::vector<double>& inputs);
		std::vector<double> calculateOutputNodeValues(std::vector<double>& expected_outputs);
		std::vector<double> calculateHiddenNodeValues(Layer& nextLayer, std::vector<double>& nextNodeValues);
		void applyGradients(double learnRate);

		void updateGradients();
		inline double relu(double x) {
			return (x > 0.0) ? x : 0.0;
		}

		inline double step(double x) {
			return (x >= 0.0) ? 1.0 : 0.0;
		}

		inline double sigmoid(double x) {
			if (x >= 0) {
				double z = std::exp(-x);
				return 1.0 / (1.0 + z);
			} else {
				double z = std::exp(x);
				return z / (1.0 + z);
			}
		}
};
