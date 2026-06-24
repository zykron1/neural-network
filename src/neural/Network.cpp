#include "Network.hpp"

std::vector<double> Network::evaluate(std::vector<double> inputs) {
    std::vector<double> outputs = inputs;

    for (auto& l : layers) {
        outputs = l.calculateOutputs(outputs);
    }

    return outputs;
}
