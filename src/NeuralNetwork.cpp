#include "NeuralNetwork.hpp"

bool NeuralNetwork::Validate(int inputs, int outputs, const std::vector<Neuron>& neurons)
{
    if (inputs == 0 || outputs == 0 || neurons.empty())
        return false;



    return false;
}

bool NeuralNetwork::Make(int inputs, int outputs, std::vector<Neuron>&& neurons)
{
    if (Validate(inputs, outputs, neurons))
    {
        m_Inputs = inputs;
        m_Outputs = outputs;
        m_Neurons = std::move(neurons);
    }
    return false;
}

bool NeuralNetwork::Evaluate(const std::vector<float>& inputs, std::vector<float>& outputs)
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
        outputs[i] = Sigmoid(sum);
    }

    return true;
}
