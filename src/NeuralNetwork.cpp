#include "NeuralNetwork.hpp"

#include "Utils.hpp"

namespace BrainFramework
{

BasicNeuralNetwork::ValidateResult BasicNeuralNetwork::Validate(int inputs, int outputs, const std::vector<Neuron>& neurons)
{
    const int neuronCount = static_cast<int>(neurons.size());

    // Format
    if (inputs == 0 || outputs == 0 || neurons.empty() || inputs + outputs > neuronCount)
        return ValidateResult::InvalidFormat;

    // Links
    for (const Neuron& neuron : neurons)
    {
        for (const Link& link : neuron.links)
        {
            if (link.neuronIndex < 0 || link.neuronIndex >= neuronCount)
            {
                return ValidateResult::InvalidLink;
            }
        }
    }

    /*
    // CyclicDependencies
    {
        std::vector<int> neuronDepths;
        neuronDepths.resize(neuronCount);

        std::queue<int> toCompute;
        for (int i = 0; i < neuronCount; ++i)
        {
            const Neuron& neuron = neurons[i];
            if (i < inputs || neuron.links.empty())
            {
                neuronDepths[i] = 0;
            }
            else
            {
                neuronDepths[i] = -1;
                toCompute.push(i);
            }
        }

        int cyclicDependencyTimeout = static_cast<int>(toCompute.size());
        while (!toCompute.empty())
        {
            int currentNeuronIndex = toCompute.front();
            toCompute.pop();

            const Neuron& currentNeuron = neurons[currentNeuronIndex];

            int depth = -1;
            bool allDetermined = true;
            for (const Link& link : currentNeuron.links)
            {
                int incomingDepth = neuronDepths[link.neuronIndex];
                if (incomingDepth >= 0)
                {
                    depth = std::max(depth, incomingDepth + 1);
                }
                else
                {
                    allDetermined = false;
                    break;
                }
            }

            if (allDetermined)
            {
                neuronDepths[currentNeuronIndex] = depth;
                cyclicDependencyTimeout = static_cast<int>(toCompute.size());
            }
            else
            {
                toCompute.push(currentNeuronIndex);
                cyclicDependencyTimeout--;
            }

            if (cyclicDependencyTimeout < 0)
            {
                return ValidateResult::InvalidCyclicDependency;
            }
        }
    }
    */

    return ValidateResult::Valid;
}

bool BasicNeuralNetwork::Make(int inputs, int outputs, std::vector<Neuron>&& neurons)
{
    if (Validate(inputs, outputs, neurons) == ValidateResult::Valid)
    {
        m_Inputs = inputs;
        m_Outputs = outputs;
        m_Neurons = std::move(neurons);
        return true;
    }
    return false;
}

bool BasicNeuralNetwork::Evaluate(const std::vector<float>& inputs, std::vector<float>& outputs)
{
    if (inputs.size() != m_Inputs || outputs.size() != m_Outputs)
    {
        return false;
    }

    const int size = static_cast<int>(m_Neurons.size());
    const int outputsStart = size - m_Outputs;

    // Fill inputs
    for (int i = 0; i < m_Inputs; ++i)
    {
        m_Neurons[i].value = inputs[i];
    }

    // Propagate
    for (int i = m_Inputs; i < outputsStart; ++i)
    {
        Neuron& neuron = m_Neurons[i];
        float sum = 0.0f;
        for (const Link& link : neuron.links)
        {
            sum += link.weight * m_Neurons[link.neuronIndex].value;
        }
        neuron.value = Sigmoid(sum);
    }
    
    // Read outputs
    for (int i = outputsStart; i < size; ++i)
    {
        Neuron& neuron = m_Neurons[i];
        float sum = 0.0f;
        for (const Link& link : neuron.links)
        {
            sum += link.weight * m_Neurons[link.neuronIndex].value;
        }
        outputs[i - outputsStart] = Sigmoid(sum);
    }

    return true;
}

LayeredNeuralNetwork::ValidateResult LayeredNeuralNetwork::Validate(const std::vector<int>& layerSizes, const std::vector<float>& weights)
{
    // Format
    if (layerSizes.size() < 2)
        return ValidateResult::InvalidFormat;

    // Weights
    int expectedWeights = 0;
    const int layers = static_cast<int>(layerSizes.size());
    for (int i = 1; i < layers; ++i)
    {
        expectedWeights += (layerSizes[i - 1] * layerSizes[i]);
    }
    if (expectedWeights != static_cast<int>(weights.size()))
        return ValidateResult::InvalidWeights;

    return ValidateResult::Valid;
}

bool LayeredNeuralNetwork::Make(const std::vector<int>& layerSizes, const std::vector<float>& weights)
{
    if (Validate(layerSizes, weights) == ValidateResult::Valid)
    {
        m_LayerSizes = layerSizes;
        m_Weights = weights;
        m_Values.resize(GetNeuronsCount());
        return true;
    }
    return false;
}

bool LayeredNeuralNetwork::Evaluate(const std::vector<float>& inputs, std::vector<float>& outputs)
{
    if (inputs.size() != GetInputsCount() || outputs.size() != GetOutputsCount())
    {
        return false;
    }

    // Fill inputs
    int i = 0;
    for (; i < m_LayerSizes[0]; ++i)
    {
        m_Values[i] = inputs[i];
    }

    // Propagate
    int previousLayerWeightBeginIndex = 0;
    int previousLayerValueBeginIndex = 0;
    const int intermediateLayers = static_cast<int>(m_LayerSizes.size() - 1);
    for (int layer = 1; layer < intermediateLayers; ++layer)
    {
        for (int iOnLayer = 0; iOnLayer < m_LayerSizes[layer]; ++iOnLayer, ++i)
        {
            float sum = 0.0f;
            for (int iOnPreviousLayer = 0; iOnPreviousLayer < m_LayerSizes[layer - 1]; ++iOnPreviousLayer)
            {
                const int weightIndex = previousLayerWeightBeginIndex + iOnPreviousLayer + iOnLayer * iOnPreviousLayer;
                const int valueIndex = previousLayerValueBeginIndex + iOnPreviousLayer;
                sum += m_Weights[weightIndex] * m_Values[valueIndex];
            }
            m_Values[i] = Sigmoid(sum);
        }
        previousLayerWeightBeginIndex += m_LayerSizes[layer] * m_LayerSizes[layer - 1];
        previousLayerValueBeginIndex += m_LayerSizes[layer - 1];
    }

    // Read outputs
    const int previousLayerIndex = static_cast<int>(m_LayerSizes.size() - 2);
    for (int iOnLayer = 0; iOnLayer < m_LayerSizes.back(); ++iOnLayer, ++i)
    {
        float sum = 0.0f;
        for (int iOnPreviousLayer = 0; iOnPreviousLayer < m_LayerSizes[previousLayerIndex]; ++iOnPreviousLayer)
        {
            const int weightIndex = previousLayerWeightBeginIndex + iOnPreviousLayer + iOnLayer * iOnPreviousLayer;
            const int valueIndex = previousLayerValueBeginIndex + iOnPreviousLayer;
            sum += m_Weights[weightIndex] * m_Values[valueIndex];
        }
        outputs[iOnLayer] = Sigmoid(sum);
    }

    return true;
}

} // namespace BrainFramework
