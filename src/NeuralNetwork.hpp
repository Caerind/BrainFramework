#pragma once

#include "Utils.hpp"

namespace BrainFramework
{

class NeuralNetwork
{
public:
    struct Link
    {
        Link(int _neuronIndex, float _weight) : neuronIndex(_neuronIndex), weight(_weight) {}
        Link(const Link& other) : neuronIndex(other.neuronIndex), weight(other.weight) {}

        int neuronIndex{ -1 };
        float weight{ 0.0f };
    };

    struct Neuron
    {
        Neuron() = default;

        std::vector<Link> links;
        float value{ 0.0f };
    };

    NeuralNetwork() = default;
    NeuralNetwork(const NeuralNetwork&) = delete;
    NeuralNetwork& operator=(const NeuralNetwork&) = delete;

    enum class ValidateResult
    {
        Valid,
        InvalidFormat,
        InvalidLink,
        InvalidCyclicDependency
    };

    static ValidateResult Validate(int inputs, int outputs, const std::vector<Neuron>& neurons);

    bool Make(int inputs, int outputs, std::vector<Neuron>&& neurons);
    bool Evaluate(const std::vector<float>& inputs, std::vector<float>& outputs);

private:
    std::vector<Neuron> m_Neurons;
    int m_Inputs{ 0 };
    int m_Outputs{ 0 };
};

} // namespace BrainFramework
