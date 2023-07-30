#pragma once
#include <vector>
#include <string>
#include <numeric>
#include <iostream>
#include <random>
#include <chrono>

inline double sigmoid(double input) {
	return 1.0 / (1.0 + std::exp(-input));
}

template<typename T>
concept arithmetic = std::integral<T> or std::floating_point<T>;

template<typename T>
	requires arithmetic<T>
class SimpleNN {
public:

	static T
		mean_squared_error(std::vector<T> expected_outputs,
			std::vector<T> observed_outputs) {
		T sum = 0;
		for (size_t i = 0; i < expected_outputs.size(); i++) {
			sum += std::pow(expected_outputs[i] - observed_outputs[i], 2);
		}
		return sum / expected_outputs.size();
	}

	static T
		mean_squared_error(std::vector<std::vector<T>> expected_outputs,
			std::vector<std::vector<T>> observed_outputs) {
		T sum = 0;
		for (size_t i = 0; i < expected_outputs.size(); i++) {
			sum += mean_squared_error(expected_outputs[i], observed_outputs[i]);
		}
		return sum / expected_outputs.size();
	}

	static T
		accuracy(std::vector<T> expected_outputs,
			std::vector<T> observed_outputs) {
		T sum = 0;
		for (size_t i = 0; i < expected_outputs.size(); i++) {
			sum += expected_outputs[i] == observed_outputs[i];
		}
		return sum / expected_outputs.size();
	}

	static T
		accuracy(std::vector<std::vector<T>> expected_outputs,
			std::vector<std::vector<T>> observed_outputs) {
		T sum = 0;
		for (size_t i = 0; i < expected_outputs.size(); i++) {
			sum += accuracy(expected_outputs[i], observed_outputs[i]);
		}
		return sum / expected_outputs.size();
	}

	static std::vector<T>
		softmax(std::vector<T> input) {
		T sum = 0;
		for (size_t i = 0; i < input.size(); i++) {
			sum += std::exp(input[i]);
		}
		std::vector<T> output(input.size());
		for (size_t i = 0; i < input.size(); i++) {
			output[i] = std::exp(input[i]) / sum;
		}
		return output;
	}

	static std::vector<T>
		softmax(std::vector<std::vector<T>> input) {
		std::vector<T> output(input.size());
		for (size_t i = 0; i < input.size(); i++) {
			output[i] = softmax(input[i]);
		}
		return output;
	}

	static std::vector<T>
		softmax_derivative(std::vector<T> input) {
		std::vector<T> output(input.size());
		for (size_t i = 0; i < input.size(); i++) {
			output[i] = input[i] * (1 - input[i]);
		}
		return output;
	}

	static std::vector<T>
		softmax_derivative(std::vector<std::vector<T>> input) {
		std::vector<T> output(input.size());
		for (size_t i = 0; i < input.size(); i++) {
			output[i] = softmax_derivative(input[i]);
		}
		return output;
	}

	static std::vector<T>
		relu(std::vector<T> input) {
		std::vector<T> output(input.size());
		for (size_t i = 0; i < input.size(); i++) {
			output[i] = input[i] > 0 ? input[i] : 0;
		}
		return output;
	}

	static std::vector<T>
		relu(std::vector<std::vector<T>> input) {
		std::vector<T> output(input.size());
		for (size_t i = 0; i < input.size(); i++) {
			output[i] = relu(input[i]);
		}
		return output;
	}

	static std::vector<T>
		relu_derivative(std::vector<T> input) {
		std::vector<T> output(input.size());
		for (size_t i = 0; i < input.size(); i++) {
			output[i] = input[i] > 0 ? 1 : 0;
		}
		return output;
	}

	static std::vector<T>
		relu_derivative(std::vector<std::vector<T>> input) {
		std::vector<T> output(input.size());
		for (size_t i = 0; i < input.size(); i++) {
			output[i] = relu_derivative(input[i]);
		}
		return output;
	}

	static std::vector<T>
		leaky_relu(std::vector<T> input, T alpha = 0.01) {
		std::vector<T> output(input.size());
		for (size_t i = 0; i < input.size(); i++) {
			output[i] = input[i] > 0 ? input[i] : alpha * input[i];
		}
		return output;
	}

	static std::vector<T>
		leaky_relu(std::vector<std::vector<T>> input, T alpha = 0.01) {
		std::vector<T> output(input.size());
		for (size_t i = 0; i < input.size(); i++) {
			output[i] = leaky_relu(input[i], alpha);
		}
		return output;
	}

	static std::vector<T>
		leaky_relu_derivative(std::vector<T> input, T alpha = 0.01) {
		std::vector<T> output(input.size());
		for (size_t i = 0; i < input.size(); i++) {
			output[i] = input[i] > 0 ? 1 : alpha;
		}
		return output;
	}

	static std::vector<T>
		leaky_relu_derivative(std::vector<std::vector<T>> input, T alpha = 0.01) {
		std::vector<T> output(input.size());
		for (size_t i = 0; i < input.size(); i++) {
			output[i] = leaky_relu_derivative(input[i], alpha);
		}
		return output;
	}

	static std::vector<T>
		identity(std::vector<T> input) {
		return input;
	}

	static std::vector<T>
		identity(std::vector<std::vector<T>> input) {
		return input;
	}

	static std::vector<T>
		identity_derivative(std::vector<T> input) {
		std::vector<T> output(input.size());
		for (size_t i = 0; i < input.size(); i++) {
			output[i] = 1;
		}
		return output;
	}

	static std::vector<T>
		identity_derivative(std::vector<std::vector<T>> input) {
		std::vector<T> output(input.size());
		for (size_t i = 0; i < input.size(); i++) {
			output[i] = identity_derivative(input[i]);
		}
		return output;
	}

	static std::vector<T>
		hyperbolic_tangent(std::vector<T> input) {
		std::vector<T> output(input.size());
		for (size_t i = 0; i < input.size(); i++) {
			output[i] = std::tanh(input[i]);
		}
		return output;
	}

	static std::vector<T>
		hyperbolic_tangent(std::vector<std::vector<T>> input) {
		std::vector  <T> output(input.size());
		for (size_t i = 0; i < input.size(); i++) {
			output[i] = hyperbolic_tangent(input[i]);
		}
		return output;
	}

	static std::vector<T>
		hyperbolic_tangent_derivative(std::vector<T> input) {
		std::vector  <T> output(input.size());
		for (size_t i = 0; i < input.size(); i++) {
			output[i] = 1 - std::pow(std::tanh(input[i]), 2);
		}
		return output;
	}

	static std::vector<T>
		hyperbolic_tangent_derivative(std::vector<std::vector<T>> input) {
		std::vector  <T> output(input.size());
		for (size_t i = 0; i < input.size(); i++) {
			output[i] = hyperbolic_tangent_derivative(input[i]);
		}
		return output;
	}

	static std::vector<T>
		softplus(std::vector<T> input) {
		std::vector  <T> output(input.size());
		for (size_t i = 0; i < input.size(); i++) {
			output[i] = std::log(1 + std::exp(input[i]));
		}
		return output;
	}

	static std::vector<T>
		softplus(std::vector<std::vector<T>> input) {
		std::vector  <T> output(input.size());
		for (size_t i = 0; i < input.size(); i++) {
			output[i] = softplus(input[i]);
		}
		return output;
	}

	static std::vector<T>
		softplus_derivative(std::vector<T> input) {
		std::vector  <T> output(input.size());
		for (size_t i = 0; i < input.size(); i++) {
			output[i] = 1 / (1 + std::exp(-input[i]));
		}
		return output;
	}

	static std::vector<T>
		softplus_derivative(std::vector<std::vector<T>> input) {
		std::vector  <T> output(input.size());
		for (size_t i = 0; i < input.size(); i++) {
			output[i] = softplus_derivative(input[i]);
		}
		return output;
	}

private:
	// https://en.cppreference.com/w/cpp/numeric/valarray
	T*** weights; // Replace with unique pointer to avoid memory leaks? https://stackoverflow.com/questions/37205179/vs2015-c6386-buffer-overrun-while-writing-even-for-same-index-value
	const size_t total_layers;
	size_t* neurons_per_layer;

	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::default_random_engine generator { seed };
	std::normal_distribution<double> distribution { 0.0, 0.5 };

public:
	SimpleNN(const std::vector<size_t> neurons_per_layer)
		: total_layers(neurons_per_layer.size()) {

		this->neurons_per_layer = new size_t[neurons_per_layer.size()];
		for (size_t i = 0; i < neurons_per_layer.size(); i++) {
			this->neurons_per_layer[i] = neurons_per_layer[i];
		}

		// There are no weights for the first layer, as this is the input layer
		size_t total_weighted_layers = this->total_layers - 1;
		this->weights = new T**[total_weighted_layers];

		for (size_t i = 0; i < total_weighted_layers; i++) {
			this->weights[i] = new T*[neurons_per_layer[i + 1]];

			for (size_t j = 0; j < this->neurons_per_layer[i + 1]; j++) {
				this->weights[i][j] = new T[neurons_per_layer[i]];
				for (size_t k = 0; k < neurons_per_layer[i]; k++) {
					this->weights[i][j][k] = this->distribution(this->generator);
				}
			}
		}
	}

	~SimpleNN() {
		for (size_t i = 0; i < this->total_layers - 1; i++) {
			delete[] this->weights[i];
		}
		delete[] this->weights;
		delete[] this->neurons_per_layer;
	}

	T train(
		size_t iterations,
		std::vector<T> inputs
	) {
		return this->train(iterations, inputs, inputs);
	}

	T train(
			size_t iterations,
			double learning_rate,
			std::vector<T> inputs,
			std::vector<T> desired_outputs
		) {
		std::vector<std::vector<T>> outputs;

		// For each iteration, run the network, find the error, and adjust weights
		for (size_t i = 0; i < iterations; i++) {
			outputs = this->predict(inputs);

			std::vector<std::vector<T>> delta_outputs = std::vector<std::vector<T>>(this->total_layers);

			// Iterate over the network layers in reverse order (ignoring the input layer).
			for (size_t layer = this->total_layers - 1; layer > 0; layer--) {
				delta_outputs[layer] = std::vector<T>(this->neurons_per_layer[layer]);

				for (size_t neuron_in_layer = 0; neuron_in_layer < this->neurons_per_layer[layer]; neuron_in_layer++) {

					double total_error_for_neuron = 0.0;
					// For the output layer, the error is the difference between the desired output and the observed output.
					if (layer == this->total_layers - 1) {
						total_error_for_neuron = desired_outputs[neuron_in_layer] - outputs[layer][neuron_in_layer];
					}
					// For the hidden layers, the error of each neuron is the product if the of the error vector and the weights connecting the neurons in the next layer that connect to this one
					else {
						for (size_t neuron_in_next_layer = 0; neuron_in_next_layer < this->neurons_per_layer[layer + 1]; neuron_in_next_layer++) {
							total_error_for_neuron += this->weights[layer][neuron_in_next_layer][neuron_in_layer] * delta_outputs[layer + 1][neuron_in_next_layer];
						}
					}

					delta_outputs[layer][neuron_in_layer] = total_error_for_neuron * outputs[layer][neuron_in_layer] * (1.0 - outputs[layer][neuron_in_layer]);
				}

				for (size_t neuron_in_layer = 0; neuron_in_layer < this->neurons_per_layer[layer]; neuron_in_layer++) {
					T weight_change;
					for (size_t neuron_in_previous_layer = 0; neuron_in_previous_layer < this->neurons_per_layer[layer - 1]; neuron_in_previous_layer++) {

						// std::cout << "Calculating Weight Change [layer = " << layer << ", neuron = " << neuron_in_layer << "]: " << "previous layer = " << layer - 1 << ", neuron in previous layer = " << neuron_in_previous_layer << std::endl;
						weight_change = learning_rate * delta_outputs[layer][neuron_in_layer] * outputs[layer - 1][neuron_in_previous_layer];

						// std::cout << "Weight Change [layer = " << layer << ", neuron = " << neuron_in_layer << "]: " << weight_change << std::endl;
						this->weights[layer - 1][neuron_in_layer][neuron_in_previous_layer] += weight_change;
					}

				}
			}

		}
		return mean_squared_error(outputs[this->total_layers - 1], desired_outputs);
	}

private:
	/* Run the given network layer for the given inputs */
	std::vector<T> run_layer(size_t layer, std::vector<T> layer_inputs) const {
		
		const size_t number_of_neurons = this->neurons_per_layer[layer];
		
		std::vector<T> weighted_layer_input(number_of_neurons);
		std::vector<T> layer_output(number_of_neurons);
		
		for (size_t i = 0; i < number_of_neurons; i++) {
			/* The total weighted input to a neuron is the product
			* of the layer input vector multiplied by the weight
			* vector for that connection from the previous layer. */
			T weighted_input = 0.0;
			for (int j = 0; j < layer_inputs.size(); j++) {
				weighted_input += layer_inputs[j] * this->weights[layer - 1][i][j];
			}
			weighted_layer_input[i] = weighted_input;
		}

		// Apply the activation function to the weighted input of each neuron.
		for (int i = 0; i < number_of_neurons; i++) {
			layer_output[i] = sigmoid(weighted_layer_input[i]);
		}

		layer_output.shrink_to_fit();
		return layer_output;
	}

public:
	/* Returns are vector of output vectors for each layer of the network.
	*  The first vector in the vector is the original input and the last vector is the final network output.
	*/
	std::vector<std::vector<T>> predict(std::vector<T> inputs) const {

		std::vector<std::vector<T>> outputs = std::vector<std::vector<T>>();
		std::vector<T> current_input = inputs;
		outputs.push_back(current_input);

		for (size_t weighted_layer = 1; weighted_layer < this->total_layers; weighted_layer++) {
			current_input = run_layer(weighted_layer, current_input);
			outputs.push_back(current_input);
		}

		return outputs;
	}
};
