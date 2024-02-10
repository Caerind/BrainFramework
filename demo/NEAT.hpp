#pragma once

#include "../src/BrainFramework.hpp"

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

    Genome(int inputs, int outputs, bool generateIOGenes)
        : m_Inputs(inputs)
        , m_Outputs(outputs)
    {
        m_MutationChances[Mutations::Connections] = k_MutateConnectionsChance;
        m_MutationChances[Mutations::Links] = k_LinkMutationChance;
        m_MutationChances[Mutations::Bias] = k_BiasMutationChance;
        m_MutationChances[Mutations::Node] = k_NodeMutationChance;
        m_MutationChances[Mutations::Enable] = k_EnableMutationChance;
        m_MutationChances[Mutations::Disable] = k_DisableMutationChance;
        m_MutationChances[Mutations::Step] = k_StepSize;

        if (generateIOGenes)
        {
            // TODO
        }
    }

    Genome(const Genome&) = delete;
    Genome& operator=(const Genome&) = delete;

    void Copy(const Genome& other)
    {
        m_Genes.clear();
        for (const Gene& gene : other.m_Genes)
        {
            m_Genes.push_back(gene);
        }

        m_MutationChances[Mutations::Connections] = other.m_MutationChances.at(Mutations::Connections);
        m_MutationChances[Mutations::Links] = other.m_MutationChances.at(Mutations::Links);
        m_MutationChances[Mutations::Bias] = other.m_MutationChances.at(Mutations::Bias);
        m_MutationChances[Mutations::Node] = other.m_MutationChances.at(Mutations::Node);
        m_MutationChances[Mutations::Enable] = other.m_MutationChances.at(Mutations::Enable);
        m_MutationChances[Mutations::Disable] = other.m_MutationChances.at(Mutations::Disable);
        m_MutationChances[Mutations::Step] = other.m_MutationChances.at(Mutations::Step);
    }

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

    bool MakeNeuralNetwork(BrainFramework::NeuralNetwork& neuralNetwork) const;

    static Genome&& Crossover(const Genome& genome1, const Genome& genome2)
    {
        // Make sure genome1 is the higher fitness genome
        if (genome2.m_Score > genome1.m_Score)
        {
            return Crossover(genome2, genome1);
        }

        Genome child(genome1.m_Inputs, genome1.m_Outputs, false);

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
    int m_Inputs;
    int m_Outputs;
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

    bool SameSpecies(const Genome& genome)
    {
        constexpr float k_DeltaDisjoint = 2.0f;
        constexpr float k_DeltaWeights = 0.4f;
        constexpr float k_DeltaThreshold = 1.0f;

        const Genome& firstGenome = m_Genomes[0];

        float deltaDisjoint = k_DeltaDisjoint * Disjoint(genome.GetGenes(), firstGenome.GetGenes());
        float deltaWeights = k_DeltaWeights * Weights(genome.GetGenes(), firstGenome.GetGenes());
        return deltaDisjoint + deltaWeights < k_DeltaThreshold;
    }

    void AddGenome(Genome&& genome)
    {
        m_Genomes.push_back(std::move(genome));
    }

    const std::vector<Genome>& GetGenomes() const { return m_Genomes; }

private:
    static float Disjoint(const std::vector<Gene>& genes1, const std::vector<Gene>& genes2)
    {
        std::unordered_set<int> innovationSet1;
        for (const Gene& gene : genes1)
            innovationSet1.insert(gene.GetInnovation());

        std::unordered_set<int> innovationSet2;
        for (const Gene& gene : genes2)
            innovationSet2.insert(gene.GetInnovation());

        int disjointGenes = 0;
        for (const Gene& gene : genes1)
        {
            if (innovationSet2.find(gene.GetInnovation()) == innovationSet2.end())
                disjointGenes++;
        }
        for (const Gene& gene : genes2)
        {
            if (innovationSet1.find(gene.GetInnovation()) == innovationSet1.end())
                disjointGenes++;
        }

        const int maxGenesCount = static_cast<int>(genes1.size() > genes2.size() ? genes1.size() : genes2.size());
        return maxGenesCount > 0 ? static_cast<float>(disjointGenes) / maxGenesCount : 0.0f;
    }

    static float Weights(const std::vector<Gene>& genes1, const std::vector<Gene>& genes2)
    {
        std::unordered_map<int, const Gene*> innovationToGene;
        for (const Gene& gene : genes2)
        {
            innovationToGene[gene.GetInnovation()] = &gene;
        }

        float weightDifferenceSum = 0.0f;
        int coincidentGenesCount = 0;

        for (const Gene& gene : genes1)
        {
            auto itr = innovationToGene.find(gene.GetInnovation());
            if (itr != innovationToGene.end())
            {
                weightDifferenceSum += std::abs(gene.GetWeight() - itr->second->GetWeight());
                coincidentGenesCount++;
            }
        }

        return coincidentGenesCount > 0 ? weightDifferenceSum / coincidentGenesCount : 0.0f;
    }

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
        for (int i = 0; i < k_Population; ++i)
        {
            Genome genome(simulation.GetInputsCount(), simulation.GetOutputsCount(), true);
            genome.Mutate();
            AddToSpecies(std::move(genome));
        }

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
            m_BestGenome.Copy(*bestGenome);
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

    void AddToSpecies(Genome&& genome)
    {
        for (Species& species : m_Species)
        {
            if (species.SameSpecies(genome))
            {
                species.AddGenome(std::move(genome));
                return;
            }
        }
        
        Species& species = m_Species.emplace_back();
        species.AddGenome(std::move(genome));
    }

private:
    Genome m_BestGenome;
    std::vector<Species> m_Species;
    int m_Generation{ 0 };
    int m_Innovation{ 0 };
    int m_CurrentSpecies{ 0 };
    int m_CurrentGenome{ 0 };

    static constexpr int k_Population = 300;
};
