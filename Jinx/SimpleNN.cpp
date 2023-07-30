#include "SimpleNN.h"
#include <iostream>

template <typename T>
void printvec(std::vector<std::vector<T>> vec, size_t min = 0, size_t max = -1) {
	size_t min_ = min > 0 ? min : 0;
	size_t max_ = vec.size();
	if (max < 0) {
		max_ = max < max_ ? max : max_;
	}
	for (size_t i = min_; i < max_; i++) {
		for (size_t j = 0; j < vec[i].size(); j++) {
			std::cout << vec[i][j] << " ";
		}
		std::cout << std::endl;
	}
}

int main() {
	//std::vector<std::vector<double>> x = {
	//	{ 1, 1 },
	//	{ 1, 0 },
	//	{ 0, 1 },
	//	{ 0, 0 }
	//};
	//std::vector<std::vector<double>> y = {
	//	{ 1 },
	//	{ 1 },
	//	{ 1 },
	//	{ 0 }
	//};
	//SimpleNN<double> nn(2, 2, 1);
	//nn.train(1000, 0.1, x, y);
	//std::vector<std::vector<double>> res = nn.predict(x);
	//printvec(res);
	//return 0;
	
	SimpleNN<double> network = SimpleNN<double>({ 3, 20, 50, 20, 3 });
	for (int i = 0; i < 1000; i++) {
		auto a = network.train(1, 0.5, { 1.0, 0.0, 0.0 }, { 1.0, 0.0, 0.0 });
		auto b = network.train(1, 0.5, { 0.0, 1.0, 0.0 }, { 0.0, 1.0, 0.0 });
		auto c = network.train(1, 0.5, { 0.0, 0.0, 1.0 }, { 0.0, 0.0, 1.0 });
	}
	
	printvec(network.predict({ 1.0, 0.0, 0.0 }), 4);
	printvec(network.predict({ 0.0, 1.0, 0.0 }), 4);
	printvec(network.predict({ 0.0, 0.0, 1.0 }), 4);
}