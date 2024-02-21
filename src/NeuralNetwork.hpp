#pragma once

#include "Utils.hpp"

namespace BrainFramework
{

class NeuralNetwork
{
public:
    NeuralNetwork() = default;
    NeuralNetwork(const NeuralNetwork&) = delete;
    NeuralNetwork& operator=(const NeuralNetwork&) = delete;

    virtual bool Evaluate(const std::vector<float>& inputs, std::vector<float>& outputs) = 0;
    virtual int GetInputsCount() = 0;
    virtual int GetOutputsCount() = 0;
    virtual int GetNeuronsCount() = 0;
    virtual int GetLinksCount() = 0;

    virtual bool LoadFromFile(const std::string& filename) = 0;
    virtual bool SaveToFile(const std::string& filename) = 0;
};

class BasicNeuralNetwork : public NeuralNetwork
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

    BasicNeuralNetwork() = default;
    BasicNeuralNetwork(const BasicNeuralNetwork&) = delete;
    BasicNeuralNetwork& operator=(const BasicNeuralNetwork&) = delete;

    enum class ValidateResult
    {
        Valid,
        InvalidFormat,
        InvalidLink,
        InvalidCyclicDependency
    };

    static ValidateResult Validate(int inputs, int outputs, const std::vector<Neuron>& neurons);
    bool Make(int inputs, int outputs, std::vector<Neuron>&& neurons);

    bool Evaluate(const std::vector<float>& inputs, std::vector<float>& outputs) override;

    int GetInputsCount() override { return m_Inputs; }
    int GetOutputsCount() override { return m_Outputs; }
    int GetNeuronsCount() override { return static_cast<int>(m_Neurons.size()); }
    int GetLinksCount() override 
    {
        int count = 0;
        for (const Neuron& neuron : m_Neurons)
            count += static_cast<int>(neuron.links.size());
        return count;
    }

    bool LoadFromFile(const std::string& filename) override
    {
        return false;
    }

    bool SaveToFile(const std::string& filename) override
    {
        return false;
    }

private:
    std::vector<Neuron> m_Neurons;
    int m_Inputs{ 0 };
    int m_Outputs{ 0 };
};

class LayeredNeuralNetwork : public NeuralNetwork
{
public:
    LayeredNeuralNetwork() = default;
    LayeredNeuralNetwork(const LayeredNeuralNetwork&) = delete;
    LayeredNeuralNetwork& operator=(const LayeredNeuralNetwork&) = delete;

    enum class ValidateResult
    {
        Valid,
        InvalidFormat,
        InvalidWeights
    };

    static ValidateResult Validate(const std::vector<int>& layerSizes, const std::vector<float>& weights);
    bool Make(const std::vector<int>& layerSizes, const std::vector<float>& weights);

    bool Evaluate(const std::vector<float>& inputs, std::vector<float>& outputs) override;

    int GetInputsCount() override { return m_LayerSizes[0]; }
    int GetOutputsCount() override { return m_LayerSizes.back(); }
    int GetNeuronsCount() override 
    { 
        int count = 0;
        for (int v : m_LayerSizes)
            count += v;
        return count;
    }
    int GetLinksCount() override
    {
        int count = 0;
        const int layers = static_cast<int>(m_LayerSizes.size());
        for (int i = 1; i < layers; ++i)
        {
            count += (m_LayerSizes[i - 1] * m_LayerSizes[i]);
        }
        return count;
    }

    bool LoadFromFile(const std::string& filename) override
    {
        std::ifstream file(filename);
        if (!file)
            return false;

        auto computeSizeToReserve = [](const std::string& str)
        {
            std::size_t size = 0;
            const std::size_t strSize = str.size();
            for (std::size_t i = 0; i < strSize; ++i)
                if (std::isspace(str[i]))
                    size++;
            return size;
        };

        std::string line;
        
        if (!std::getline(file, line))
            return false;
        m_LayerSizes.reserve(computeSizeToReserve(line));
        std::stringstream layersStream(line);
	    int layersValue;
        while (layersStream >> layersValue)
            m_LayerSizes.push_back(layersValue);

        if (!std::getline(file, line))
            return false;
        m_Weights.reserve(computeSizeToReserve(line));
        std::stringstream weightsStream(line);
	    float weightsValue;
        while (weightsStream >> weightsValue)
            m_Weights.push_back(weightsValue);
        
        file.close();
        return true;
    }

    bool SaveToFile(const std::string& filename) override
    {
        std::ofstream file(filename);
        if (!file)
            return false;
        
        for (int s : m_LayerSizes)
            file << s << " ";
        file << std::endl;
        
        for (float w : m_Weights)
            file << w << " ";
        file << std::endl;
        
        file.close();
        return true;
    }

private:
    std::vector<float> m_Values;
    std::vector<int> m_LayerSizes;
    std::vector<float> m_Weights;
};

} // namespace BrainFramework
