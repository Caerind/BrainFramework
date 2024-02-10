#pragma once

#include "../src/BrainFramework.hpp"

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
        m_MutationChances[Mutations::Connections] = k_MutateConnectionsChance;
        m_MutationChances[Mutations::Links] = k_LinkMutationChance;
        m_MutationChances[Mutations::Bias] = k_BiasMutationChance;
        m_MutationChances[Mutations::Node] = k_NodeMutationChance;
        m_MutationChances[Mutations::Enable] = k_EnableMutationChance;
        m_MutationChances[Mutations::Disable] = k_DisableMutationChance;
        m_MutationChances[Mutations::Step] = k_StepSize;
    }

    //Genome(const Genome&) = delete;
    //Genome& operator=(const Genome&) = delete;

    void CopyFrom(const Genome& other)
    {
        m_Inputs = other.m_Inputs;
        m_Outputs = other.m_Outputs;

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

    void Initialize(int inputs, int outputs)
    {
        m_Inputs = inputs;
        m_Outputs = outputs;
        // TODO : IO Genes
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

        // TODO
    }

    bool MakeNeuralNetwork(BrainFramework::NeuralNetwork& neuralNetwork) const;

    void UpdateGlobalRank(int globalRank) // Ranks are inverted
    {
        m_GlobalRank = globalRank;
    }

    void SetScore(float score)
    {
        m_Score = score;
    }

    void Crossover(const Genome& genome1, const Genome& genome2)
    {
        // Make sure genome1 is the higher fitness genome
        if (genome2.m_Score > genome1.m_Score)
        {
            Crossover(genome2, genome1);
            return;
        }

        m_Inputs = genome1.m_Inputs;
        m_Outputs = genome1.m_Outputs;

        std::unordered_map<int, Gene> innovations2;
        for (const Gene& gene : genome2.m_Genes)
        {
            innovations2[gene.GetInnovation()] = gene;
        }

        for (const Gene& gene1 : genome1.m_Genes)
        {
            auto it = innovations2.find(gene1.GetInnovation());
            const Gene* gene = (it != innovations2.end() && (rand() % 2 == 0) && it->second.IsEnabled()) ? &(it->second) : &gene1;
            m_Genes.push_back(*gene);
        }

        for (const auto& mutationChance : genome1.m_MutationChances)
        {
            m_MutationChances[mutationChance.first] = mutationChance.second;
        }
    }

    const std::vector<Gene>& GetGenes() const { return m_Genes; }
    const std::unordered_map<Mutations, float>& GetMutationChances() const { return m_MutationChances; }
    int GetInputs() const { return m_Inputs; }
    int GetOutputs() const { return m_Outputs; }
    int GetGlobalRank() const { return m_GlobalRank; }
    float GetScore() const { return m_Score; }

private:
    std::vector<Gene> m_Genes;
    std::unordered_map<Mutations, float> m_MutationChances;
    int m_Inputs;
    int m_Outputs;
    int m_GlobalRank{ 0 };
    float m_Score{ 0.0f };

    static constexpr float k_PerturbChance = 0.90f;
    static constexpr float k_MutateConnectionsChance = 0.25f;
    static constexpr float k_LinkMutationChance = 2.0f;
    static constexpr float k_BiasMutationChance = 0.40f;
    static constexpr float k_NodeMutationChance = 0.50f;
    static constexpr float k_EnableMutationChance = 0.2f;
    static constexpr float k_DisableMutationChance = 0.4f;
    static constexpr float k_StepSize = 0.1f;
};

class Species
{
public:
    Species() = default;
    //Species(const Species&) = delete;
    //Species& operator=(const Species&) = delete;

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

    void Update()
    {
        SortGenomes();

        if (m_Genomes[0].GetScore() > m_TopScore)
        {
            m_TopScore = m_Genomes[0].GetScore();
            m_Staleness = 0;
        }
        else
        {
            m_Staleness++;
        }
    }

    void CullSpecies(bool cutToOne)
    {
        SortGenomes();

        const std::size_t remaining = cutToOne ? 1 : static_cast<std::size_t>(std::ceil(m_Genomes.size() / 2));
        m_Genomes.erase(m_Genomes.begin() + remaining, m_Genomes.end());
    }

    void BreedChild(Genome& child) const
    {
        const int index1 = BrainFramework::RandomIndex(m_Genomes);
        const Genome& genome1 = m_Genomes[index1];

        constexpr float k_CrossoverChance = 0.75f;
        if (BrainFramework::RandomFloat() < k_CrossoverChance)
        {
            const int index2 = BrainFramework::RandomIndex(m_Genomes);
            const Genome& genome2 = m_Genomes[index2];

            child.Crossover(genome1, genome2);
        }
        else
        {
            child.CopyFrom(genome1);
        }

        child.Mutate();
    }

    float CalculateAverageFitness()
    {
        float total = 0.0f;
        for (const Genome& genome : m_Genomes)
        {
            total += genome.GetGlobalRank();
        }
        m_AverageFitness = total / static_cast<float>(m_Genomes.size());
        return m_AverageFitness;
    }

    std::vector<Genome>& GetGenomes() { return m_Genomes; }
    const std::vector<Genome>& GetGenomes() const { return m_Genomes; }
    float GetTopScore() const { return m_TopScore; }
    float GetAverageFitness() const { return m_AverageFitness; }
    int GetStaleness() const { return m_Staleness; }

private:
    void SortGenomes()
    {
        std::sort(m_Genomes.begin(), m_Genomes.end(), [](const Genome& a, const Genome& b) {
            return a.GetScore() > b.GetScore();
        });
    }

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
    float m_AverageFitness{ 0.0f };
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
        if (m_Species.empty())
        {
            for (int i = 0; i < k_Population; ++i)
            {
                Genome genome;
                genome.Initialize(simulation.GetInputsCount(), simulation.GetOutputsCount());
                genome.Mutate();
                AddToSpecies(std::move(genome));
            }
        }

        m_CurrentSpecies = 0;
        m_CurrentGenome = 0;
        return Model::StartTraining(simulation);
    }

    void Train(BrainFramework::Simulation& simulation) override
    {
        Genome& genome = m_Species[m_CurrentSpecies].GetGenomes()[m_CurrentGenome];

        BrainFramework::NeuralNetwork neuralNetwork;
        genome.MakeNeuralNetwork(neuralNetwork);

        if (simulation.GetResult() == BrainFramework::Simulation::Result::None)
        {
            simulation.Initialize(neuralNetwork);
        }

        auto result = BrainFramework::Simulation::Result::Initialized;
        do 
        {
            result = simulation.Step(neuralNetwork);
            genome.SetScore(simulation.GetScore());
        } while (result == BrainFramework::Simulation::Result::Ongoing);

        NextGenome();
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
            m_BestGenome.CopyFrom(*bestGenome);
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

    void NextGenome()
    {
        m_CurrentGenome++;
        if (m_CurrentGenome >= static_cast<int>(m_Species[m_CurrentSpecies].GetGenomes().size()))
        {
            m_CurrentGenome = 0;
            m_CurrentSpecies++;
            if (m_CurrentSpecies >= static_cast<int>(m_Species.size()))
            {
                m_CurrentSpecies = 0;
                NewGeneration();
            }
        }
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

    void NewGeneration()
    {
        for (Species& species : m_Species)
            species.Update();

        // Cull the bottom half of each species
        for (Species& species : m_Species)
            species.CullSpecies(false);

        // Remove Stale Species
        constexpr int k_StaleSpecies = 15;
        for (int i = static_cast<int>(m_Species.size()) - 1; i >= 0; --i)
        {
            const Species& species = m_Species[i];
            if (species.GetStaleness() >= k_StaleSpecies && species.GetTopScore() < m_MaxScore)
            {
                m_Species.erase(m_Species.begin() + i);
            }
        }

        // Remove Weak Species
        RankGlobally();
        float totalAverageFitness = 0.0f;
        for (Species& species : m_Species)
            totalAverageFitness += species.CalculateAverageFitness();
        for (int i = static_cast<int>(m_Species.size()) - 1; i >= 0; --i)
        {
            const Species& species = m_Species[i];
            const int breedCount = static_cast<int>(std::floor(species.GetAverageFitness() / totalAverageFitness * k_Population));
            if (breedCount <= 0)
            {
                m_Species.erase(m_Species.begin() + i);
            }
        }

        // Breed from the best
        RankGlobally();
        totalAverageFitness = 0.0f;
        for (Species& species : m_Species)
            totalAverageFitness += species.CalculateAverageFitness();
        std::vector<Genome> children;
        for (const Species& species : m_Species)
        {
            const int breedCount = static_cast<int>(std::floor(species.GetAverageFitness() / totalAverageFitness * k_Population) - 1);
            for (int i = 0; i < breedCount; ++i)
            {
                Genome& child = children.emplace_back();
                species.BreedChild(child);
            }
        }

        // Cull all but the top member of each species
        for (Species& species : m_Species)
            species.CullSpecies(true);

        // Complete from the very best
        while (static_cast<int>(children.size() + m_Species.size()) < k_Population)
        {
            const int speciesIndex = BrainFramework::RandomIndex(m_Species);
            Genome& child = children.emplace_back();
            m_Species[speciesIndex].BreedChild(child);
        }

        // Add to the species
        for (Genome& genome : children)
        {
            AddToSpecies(std::move(genome));
        }

        m_Generation++;
    }

    void RankGlobally()
    {
        std::vector<Genome*> allGenomes;
        allGenomes.reserve(k_Population);
        for (Species& species : m_Species)
        {
            for (Genome& genome : species.GetGenomes())
            {
                allGenomes.push_back(&genome);
            }
        }

        std::sort(allGenomes.begin(), allGenomes.end(), [](const Genome* a, const Genome* b) {
            return a->GetScore() < b->GetScore();
        });

        const int allGenomesCount = static_cast<int>(allGenomes.size());
        for (int i = 0; i < allGenomesCount; ++i)
        {
            allGenomes[i]->UpdateGlobalRank(i + 1);
        }
    }

private:
    Genome m_BestGenome;
    std::vector<Species> m_Species;
    float m_MaxScore{ 0.0f };
    int m_Generation{ 0 };
    int m_Innovation{ 0 };
    int m_CurrentSpecies{ 0 };
    int m_CurrentGenome{ 0 };

    static constexpr int k_Population = 300;
};
