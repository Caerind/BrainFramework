#include "NEAT.hpp"

bool Genome::MakeNeuralNetwork(BrainFramework::NeuralNetwork& neuralNetwork) const
{
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

    /*
    BakingNetwork bakingNeurons;

    PrebakeNetwork(genome, bakingNeurons);
    ComputeNeuronsDepth(bakingNeurons);

    // Bake
    m_Neurons.resize(bakingNeurons.size());
    m_OutputsStart = 0;
    // TODO


    void NeuralNetwork::PrebakeNetwork(const Genome& genome, NeuralNetwork::BakingNetwork& bakingNeurons)
    {
        bakingNeurons.reserve(k_Outputs + k_Inputs); // TODO : Improve reserve based on Genome genes count

        for (int i = 0; i < k_Inputs; ++i)
        {
            bakingNeurons[i].value = 0;
        }

        for (int o = 0; o < k_Outputs; ++o)
        {
            bakingNeurons[o].value = 0; // TODO : o
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
    */

    return false;
}