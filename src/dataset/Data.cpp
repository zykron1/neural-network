#include "Data.hpp"

void Data::generateData() {
	for (int i = 0; i < 200; i++) {
        double x = dist(gen);
        double y = dist(gen);

        if (y > x) {
            x_red.push_back(x);
            y_red.push_back(y);
        } else {
            x_blue.push_back(x);
            y_blue.push_back(y);
        }
    }
}
