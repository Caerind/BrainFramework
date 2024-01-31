#include "NeuralNetwork.hpp"

#include "BrainFramework.hpp" // Genome/Gene

void NeuralNetwork::Generate(const Genome& genome)
{
    BakingNetwork bakingNeurons;

    PrebakeNetwork(genome, bakingNeurons);
    ComputeNeuronsDepth(bakingNeurons);

    // Bake
    m_Neurons.resize(bakingNeurons.size());
    m_OutputsStart = 0;
    // TODO
}

bool NeuralNetwork::Evaluate(const std::vector<float>& inputs, std::vector<float>& outputs)
{
    if (inputs.size() != k_Inputs || outputs.size() != k_Outputs)
    {
        return false;
    }

    // Fill inputs
    for (int i = 0; i < k_Inputs; ++i)
    {
        m_Neurons[i].value = inputs[i];
    }

    // Propagate
    for (int i = k_Inputs; i < m_OutputsStart; ++i)
    {
        Neuron& neuron = m_Neurons[i];
        float sum = 0.0f;
        for (const Neuron* incoming : neuron.incomings)
        {
            sum += incoming->weight * incoming->value;
        }
        neuron.value = Sigmoid(sum);
    }
    
    // Read outputs
    const int size = static_cast<int>(m_Neurons.size());
    for (int i = m_OutputsStart; i < size; ++i)
    {
        Neuron& neuron = m_Neurons[i];
        float sum = 0.0f;
        for (const Neuron* incoming : neuron.incomings)
        {
            sum += incoming->weight * incoming->value;
        }
        outputs[i] = Sigmoid(sum);
    }
}

void NeuralNetwork::PrebakeNetwork(const Genome& genome, NeuralNetwork::BakingNetwork& bakingNeurons)
{
    bakingNeurons.reserve(k_Outputs + k_Inputs); // TODO : Improve reserve based on Genome genes count

    for (int i = 0; i < k_Inputs; ++i)
    {
        bakingNeurons[i].value = 0;
    }

    for (int o = 0; o < k_Outputs; ++o)
    {
        bakingNeurons[o].value = 0;
    }

    for (const Gene& gene : genome.GetGenes())
    {
        if (gene.IsEnabled())
        {
            int geneIn = gene.GetIn();
            int geneOut = gene.GetOut();

            auto itrOut = bakingNeurons.find(geneOut);
            if (itrOut == bakingNeurons.end())
            {
                bakingNeurons[geneOut].value = 0;
            }

            BakingNeuron& neuronOut = bakingNeurons[geneOut];
            neuronOut.weight = gene.GetWeight();
            neuronOut.incomings.push_back(geneIn);

            auto itrIn = bakingNeurons.find(geneIn);
            if (itrIn == bakingNeurons.end())
            {
                bakingNeurons[geneIn].value = 0;
            }
        }
    }
}

void NeuralNetwork::ComputeNeuronsDepth(BakingNetwork& bakingNeurons)
{
    std::queue<int> toCompute;
    for (auto& entry : bakingNeurons)
    {
        if (entry.second.incomings.empty())
        {
            entry.second.incomingDepth = 0;
        }
        else
        {
            entry.second.incomingDepth = -1;
            toCompute.push(entry.first);
        }
    }

    while (!toCompute.empty())
    {
        int current = toCompute.front();
        toCompute.pop();

        BakingNeuron& currentNeuron = bakingNeurons[current];

        int depth = -1;
        bool allDetermined = true;
        for (int incomingNeuron : currentNeuron.incomings)
        {
            int incomingDepth = bakingNeurons[incomingNeuron].incomingDepth;
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
            currentNeuron.incomingDepth = depth;
        }
        else
        {
            toCompute.push(current);
        }
    }
}
