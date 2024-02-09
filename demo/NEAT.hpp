#pragma once

#include "../src/BrainFramework.hpp"

constexpr int k_Population = 300;
constexpr float k_DeltaDisjoint = 2.0f;
constexpr float k_DeltaWeights = 0.4f;
constexpr float k_DeltaThreshold = 1.0f;
constexpr int k_StaleSpecies = 15;
constexpr float k_TimeoutConstant = 20.0f;

constexpr float k_PerturbChance = 0.90f;
constexpr float k_CrossoverChance = 0.75f;
constexpr float k_MutateConnectionsChance = 0.25f;
constexpr float k_LinkMutationChance = 2.0f;
constexpr float k_BiasMutationChance = 0.40f;
constexpr float k_NodeMutationChance = 0.50f;
constexpr float k_EnableMutationChance = 0.2f;
constexpr float k_DisableMutationChance = 0.4f;
constexpr float k_StepSize = 0.1f;

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

    void Copy(Genome& other) const
    {
        other.m_Genes.clear();
        for (const Gene& gene : m_Genes)
        {
            other.m_Genes.push_back(gene);
        }

        other.m_MutationChances[Mutations::Connections] = m_MutationChances.at(Mutations::Connections);
        other.m_MutationChances[Mutations::Links] = m_MutationChances.at(Mutations::Links);
        other.m_MutationChances[Mutations::Bias] = m_MutationChances.at(Mutations::Bias);
        other.m_MutationChances[Mutations::Node] = m_MutationChances.at(Mutations::Node);
        other.m_MutationChances[Mutations::Enable] = m_MutationChances.at(Mutations::Enable);
        other.m_MutationChances[Mutations::Disable] = m_MutationChances.at(Mutations::Disable);
        other.m_MutationChances[Mutations::Step] = m_MutationChances.at(Mutations::Step);
    }

    bool MakeNeuralNetwork(BrainFramework::NeuralNetwork& neuralNetwork) const;

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

    static Genome&& Crossover(const Genome& genome1, const Genome& genome2)
    {
        // Make sure genome1 is the higher fitness genome
        if (genome2.m_Score > genome1.m_Score)
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

        return std::move(child);
    }

    const std::vector<Gene>& GetGenes() const { return m_Genes; }
    const std::unordered_map<Mutations, float>& GetMutationChances() const { return m_MutationChances; }
    float GetScore() const { return m_Score; }
    float GetAdjustedScore() const { return m_AdjustedScore; }

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
    float m_Score{ 0 };
    float m_AdjustedScore{ 0 };
};

class Species
{
public:
    Species() = default;
    Species(const Species&) = delete;
    Species& operator=(const Species&) = delete;

    const std::vector<Genome>& GetGenomes() const { return m_Genomes; }

private:
    std::vector<Genome> m_Genomes;
    float m_TopScore{ 0.0f };
    float m_AverageScore{ 0.0f };
    int m_Staleness{ 0 };
};

class NEAT : public BrainFramework::Model
{
public:
    NEAT() = default;
    NEAT(const NEAT&) = delete;
    NEAT& operator=(const NEAT&) = delete;

    const char* GetName() const override { return "NEAT"; }

    bool StartTraining(const BrainFramework::Simulation& simulation) override
    {
        // TODO
        m_CurrentSpecies = 0;
        m_CurrentGenome = 0;
        return Model::StartTraining(simulation);
    }

    void Train(BrainFramework::Simulation& simulation) override
    {
        // TODO
    }

    bool StopTraining(const BrainFramework::Simulation& simulation) override
    {
        const Genome* bestGenome = nullptr;
        for (const Species& species : m_Species)
        {
            for (const Genome& genome : species.GetGenomes())
            {
                if (bestGenome == nullptr || genome.GetScore() > bestGenome->GetScore())
                {
                    bestGenome = &genome;
                }
            }
        }

        Model::StopTraining(simulation);
        
        if (bestGenome != nullptr)
        {
            bestGenome->Copy(m_BestGenome);
            return true;
        }
        else
        {
            return false;
        }
    }

    bool MakeBestNeuralNetwork(BrainFramework::NeuralNetwork& neuralNetwork) override
    {
        if (IsTraining())
            return false;
        
        return m_BestGenome.MakeNeuralNetwork(neuralNetwork);
    }

private:
    Genome m_BestGenome;
    std::vector<Species> m_Species;
    int m_Generation{ 0 };
    int m_Innovation{ 0 };
    int m_CurrentSpecies{ 0 };
    int m_CurrentGenome{ 0 };
};
