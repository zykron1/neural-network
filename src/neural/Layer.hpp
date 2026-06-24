#pragma once

#include <vector>

class Layer {
	public:
		std::size_t nodes_in, nodes_out;
		std::vector<double> weights;
		std::vector<double> weightGradients;
		std::vector<double> biases;
		std::vector<double> biasGradients;

		Layer(std::size_t in, std::size_t out) :
			nodes_in(in),
			nodes_out(out),
			weights(in*out),
			weightGradients(in*out),
			biases(in),
			biasGradients(in)
		{} 

		double& weight(std::size_t in_node, std::size_t out_node);
		double& weightGradient(std::size_t in_node, std::size_t out_node);

		std::vector<double> calculateOutputs(std::vector<double>& inputs);

		void applyGradients(double learnRate);

};
