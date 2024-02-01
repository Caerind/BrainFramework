#pragma once

#include "Utils.hpp"

class Genome;

class NeuralNetwork
{
public:
    struct Link
    {
        int neuronIndex;
        float weight;
    };

    struct Neuron
    {
        Neuron() = default;
        Neuron(const Neuron&) = delete;
        Neuron& operator=(const Neuron&) = delete;

        std::vector<Link> links;
        float value{ 0.0f };
    };

    NeuralNetwork() = default;

    NeuralNetwork(const NeuralNetwork&) = delete;
    NeuralNetwork& operator=(const NeuralNetwork&) = delete;

    static bool Validate(int inputs, int outputs, const std::vector<Neuron>& neurons);

    bool Make(int inputs, int outputs, std::vector<Neuron>&& neurons);
    bool Evaluate(const std::vector<float>& inputs, std::vector<float>& outputs);

private:
    std::vector<Neuron> m_Neurons;
    int m_Inputs{ 0 };
    int m_Outputs{ 0 };
};
