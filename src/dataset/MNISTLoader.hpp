#pragma once

#include <string>
#include <utility>
#include <vector>

class MNISTLoader {
	public:
		std::string fname;

		MNISTLoader(std::string filename) : fname(filename) {};

		std::pair<std::vector<std::vector<double>>, std::vector<double>> load();
};
