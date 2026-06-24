#pragma once

#include <vector>

class Layer {
	public:
		std::size_t nodes_in, nodes_out;
		std::vector<double> weights;
		std::vector<double> biases;

		Layer(std::size_t in, std::size_t out) :
			nodes_in(in),
			nodes_out(out),
			weights(in*out),
			biases(in) 
		{} 

		double& weight(std::size_t in_node, std::size_t out_node);

		std::vector<double> calculateOutputs(std::vector<double>& inputs);

};
