#pragma once

#include "Utils.hpp"
#include "NeuralNetwork.hpp"

class Gene
{
public:
    Gene() = default;

    float GetWeight() const { return m_Weight; }
    int GetIn() const { return m_In; }
    int GetOut() const { return m_Out; }
    bool IsEnabled() const { return m_Enabled; }
    int GetInnovation() const { return m_Innovation; }

private:
    float m_Weight{ 0.0f };
    int m_In{ 0 };
    int m_Out{ 0 };
    int m_Innovation{ 0 };
    bool m_Enabled{ true };
};

class Genome
{
public:
    enum class Mutations
    {
        Connections,
        Links,
        Bias,
        Node,
        Enable,
        Disable,
        Step
    };

    Genome()
    {
        InitMutationChances();
    }

    Genome(const Genome&) = delete;
    Genome& operator=(const Genome&) = delete;

    Genome&& Copy() const
    {
        Genome other;
        for (const Gene& gene : m_Genes)
        {
            other.m_Genes.push_back(gene.Copy());
        }

        other.m_MutationChances[Mutations::Connections] = m_MutationChances.at(Mutations::Connections);
        other.m_MutationChances[Mutations::Links] = m_MutationChances.at(Mutations::Links);
        other.m_MutationChances[Mutations::Bias] = m_MutationChances.at(Mutations::Bias);
        other.m_MutationChances[Mutations::Node] = m_MutationChances.at(Mutations::Node);
        other.m_MutationChances[Mutations::Enable] = m_MutationChances.at(Mutations::Enable);
        other.m_MutationChances[Mutations::Disable] = m_MutationChances.at(Mutations::Disable);
        other.m_MutationChances[Mutations::Step] = m_MutationChances.at(Mutations::Step);

        other.m_MaxNeurons = m_MaxNeurons;

        return std::move(other);
    }

    bool Generate(NeuralNetwork& neuralNetwork);

    void Mutate()
    {
        // Alterate mutation chances
        for (auto& mutationChance : m_MutationChances) 
        {
            mutationChance.second *= (rand() % 2 == 0) ? 0.95f : 1.05263f;
        }


        // Sort genes
        std::sort(m_Genes.begin(), m_Genes.end(), [](const Gene& a, const Gene& b) {
            return a.GetOut() < b.GetOut();
        });
    }

    const std::vector<Gene>& GetGenes() const { return m_Genes; }

    static Genome&& Crossover(const Genome& genome1, const Genome& genome2)
    {
        // Make sure genome1 is the higher fitness genome
        if (genome2.m_Fitness > genome1.m_Fitness)
        {
            return Crossover(genome2, genome1);
        }

        Genome child;

        std::unordered_map<int, Gene> innovations2;
        for (const Gene& gene : genome2.m_Genes) 
        {
            innovations2[gene.GetInnovation()] = gene;
        }

        for (const Gene& gene1 : genome1.m_Genes)
        {
            auto it = innovations2.find(gene1.GetInnovation());
            const Gene* gene = (it != innovations2.end() && (rand() % 2 == 0) && it->second.IsEnabled()) ? &(it->second) : &gene1;
            child.m_Genes.push_back(*gene);
        }

        for (const auto& mutationChance : genome1.m_MutationChances)
        {
            child.m_MutationChances[mutationChance.first] = mutationChance.second;
        }

        child.m_MaxNeurons = std::max(genome1.m_MaxNeurons, genome2.m_MaxNeurons);

        return std::move(child);
    }

private:
    void InitMutationChances()
    {
        m_MutationChances[Mutations::Connections] = k_MutateConnectionsChance;
        m_MutationChances[Mutations::Links] = k_LinkMutationChance;
        m_MutationChances[Mutations::Bias] = k_BiasMutationChance;
        m_MutationChances[Mutations::Node] = k_NodeMutationChance;
        m_MutationChances[Mutations::Enable] = k_EnableMutationChance;
        m_MutationChances[Mutations::Disable] = k_DisableMutationChance;
        m_MutationChances[Mutations::Step] = k_StepSize;
    }

private:
    std::vector<Gene> m_Genes;
    std::unordered_map<Mutations, float> m_MutationChances;
    int m_MaxNeurons{ 0 }; // Not here ?

    int m_Fitness{ 0 };
    int m_AdjustedFitness{ 0 };
    int m_GlobalRank{ 0 }; // Not here ?
};

class Species
{
public:
    Species() = default;

    Species(const Species&) = delete;
    Species& operator=(const Species&) = delete;


private:
    std::vector<Genome> m_Genomes;
    int m_TopFitness{ 0 };
    int m_Staleness{ 0 };
    int m_AverageFitness{ 0 };
};

class Pool
{
public:
    Pool() = default;

    Pool(const Pool&) = delete;
    Pool& operator=(const Pool&) = delete;

private:
    std::vector<Species> m_Species;
    int m_Generation{ 0 };
    int m_Innovation{ 0 }; // Outputs (button) count ?
    int m_CurrentSpecies{ 0 };
    int m_CurrentGenome{ 0 };
    int m_CurrentFrame{ 0 };
    int m_MaxFitness{ 0 };
};