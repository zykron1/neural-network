#include "MNISTLoader.hpp"
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <sstream>
#include <string>
#include <utility>

std::vector<std::string> split(const std::string& str, char delim) {
    std::vector<std::string> tokens;
    std::stringstream ss(str);
    std::string token;

    while (std::getline(ss, token, delim)) {
        tokens.push_back(token);
    }

    return tokens;
}

std::pair<std::vector<std::vector<double>>, std::vector<double>> MNISTLoader::load() {
	std::fstream file(this->fname);
	
	std::vector<std::vector<double>> data;
	std::vector<double> labels;

	if (!file.is_open()) {
		std::cout << "failed to load file: " << this->fname << std::endl;
		throw std::runtime_error("failed to open mnist file");
	}

	std::string line;
	std::getline(file, line); // skip the first line of csv(header)

	while (std::getline(file, line)) {
		auto values = split(line, ',');
		std::vector<double> image;

		for (int i = 1; i < values.size(); i++) {
			auto value = values[i];
			image.push_back(std::stod(value) / 255);
		}
		data.push_back(image);
		labels.push_back(std::stod(values[0]));
	}

	std::cout << "successfully loaded: " << this->fname << " with " << labels.size() << " lines!" << std::endl;
	return std::make_pair(data, labels);
}
