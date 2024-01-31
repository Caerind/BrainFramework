#pragma once

#include "Utils.hpp"

class Genome;

class NeuralNetwork
{
public:
    NeuralNetwork() = default;

    NeuralNetwork(const NeuralNetwork&) = delete;
    NeuralNetwork& operator=(const NeuralNetwork&) = delete;

    void Generate(const Genome& genome);

    bool Evaluate(const std::vector<int>& inputs, std::vector<int>& outputs);

private:
    struct BakingNeuron
    {
        BakingNeuron() = default;
        BakingNeuron(const BakingNeuron&) = delete;
        BakingNeuron& operator=(const BakingNeuron&) = delete;

        std::vector<int> incomings;
        float value{ 0.0f };
        float weight{ 0.0f };
        int incomingDepth{ 0 };
        bool isInput{ false };
        bool isOutput{ false };
    };

    struct Neuron
    {
        Neuron() = default;
        Neuron(const Neuron&) = delete;
        Neuron& operator=(const Neuron&) = delete;

        std::vector<Neuron*> incomings;
        float value{ 0.0f };
        float weight{ 0.0f };
    };

    using BakingNetwork = std::unordered_map<int, BakingNeuron>;

    void PrebakeNetwork(const Genome& genome, BakingNetwork& bakingNeurons);
    void ComputeNeuronsDepth(BakingNetwork& bakingNeurons);

private:
    std::vector<Neuron> m_Neurons;
    int m_OutputsStart{ 0 };
};
