#pragma once

#include "../src/BrainFramework.hpp"

namespace NEAT
{

class Gene
{
public:
    Gene() = default;
    Gene(int in, int out, float weight, bool enabled, int innovation)
        : m_Weight(weight)
        , m_In(in)
        , m_Out(out)
        , m_Innovation(innovation)
        , m_Enabled(enabled)
    {
    }

    void CopyFrom(const Gene& other)
    {
        m_Weight = other.m_Weight;
        m_In = other.m_In;
        m_Out = other.m_Out;
        m_Innovation = other.m_Innovation;
        m_Enabled = other.m_Enabled;
    }

    void SetIn(int in) { m_In = in; }
    void SetOut(int out) { m_Out = out; }
    void SetWeight(float weight) { m_Weight = weight; }
    void SetEnabled(bool enabled) { m_Enabled = enabled; }
    void SetInnovation(int innovation) { m_Innovation = innovation; }

    int GetIn() const { return m_In; }
    int GetOut() const { return m_Out; }
    float GetWeight() const { return m_Weight; }
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
        m_MaxNeurons = other.m_MaxNeurons;

        m_Genes.clear();
        m_Genes.reserve(other.m_Genes.size());
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

    void Crossover(const Genome& genome1, const Genome& genome2)
    {
        // Make sure genome1 is the higher fitness genome
        if (genome2.m_Score > genome1.m_Score)
        {
            Crossover(genome2, genome1);
            return;
        }

        assert(genome1.m_Inputs == genome2.m_Inputs);
        assert(genome1.m_Outputs == genome2.m_Outputs);

        m_Inputs = genome1.m_Inputs;
        m_Outputs = genome1.m_Outputs;
        m_MaxNeurons = (genome1.m_MaxNeurons > genome2.m_MaxNeurons) ? genome1.m_MaxNeurons : genome2.m_MaxNeurons;

        std::unordered_map<int, Gene> innovations2;
        innovations2.reserve(genome2.m_Genes.size());
        for (const Gene& gene : genome2.m_Genes)
        {
            innovations2[gene.GetInnovation()] = gene;
        }

        m_Genes.clear();
        m_Genes.reserve(genome1.m_Genes.size());
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

    void Initialize(int inputs, int outputs)
    {
        m_Inputs = inputs;
        m_Outputs = outputs;
        m_MaxNeurons = m_Inputs + m_Outputs;
    }

    void Mutate()
    {
        // Alterate mutation chances
        for (auto& mutationChance : m_MutationChances)
        {
            mutationChance.second *= (rand() % 2 == 0) ? 0.95f : 1.05263f;
        }

        if (BrainFramework::RandomFloat() < m_MutationChances[Mutations::Connections])
            PointMutate();

        float p = static_cast<float>(m_MutationChances[Mutations::Links]);
        while (p > 0.0f)
        {
            if (BrainFramework::RandomFloat() < p)
                LinkMutate(false);
            p -= 1.0f;
        }
        
        p = static_cast<float>(m_MutationChances[Mutations::Bias]);
        while (p > 0.0f)
        {
            if (BrainFramework::RandomFloat() < p)
                LinkMutate(true);
            p -= 1.0f;
        }

        p = static_cast<float>(m_MutationChances[Mutations::Node]);
        while (p > 0.0f)
        {
            if (BrainFramework::RandomFloat() < p)
                NodeMutate();
            p -= 1.0f;
        }

        p = static_cast<float>(m_MutationChances[Mutations::Enable]);
        while (p > 0.0f)
        {
            if (BrainFramework::RandomFloat() < p)
                EnableDisableMutate(true);
            p -= 1.0f;
        }

        p = static_cast<float>(m_MutationChances[Mutations::Disable]);
        while (p > 0.0f)
        {
            if (BrainFramework::RandomFloat() < p)
                EnableDisableMutate(false);
            p -= 1.0f;
        }
    }

    bool MakeNeuralNetwork(BrainFramework::NeuralNetwork& neuralNetwork) const
    {
        std::vector<BrainFramework::NeuralNetwork::Neuron> neurons;
        neurons.reserve(m_MaxNeurons);

        // Inputs
        for (int i = 0; i < m_Inputs; ++i)
            neurons.emplace_back();

        // Middles
        const int middleNeurons = m_MaxNeurons - m_Inputs - m_Outputs;
        for (int i = 0; i < middleNeurons; ++i)
            neurons.emplace_back();

        // Outputs
        for (int i = 0; i < m_Outputs; ++i)
            neurons.emplace_back();

        // Links from Genes
        auto translateIndex = [&](int geneIndex)
        {
            if (geneIndex < m_Inputs)
                return geneIndex;
            else if (geneIndex < m_Inputs + m_Outputs)
                return geneIndex + middleNeurons;
            return geneIndex - m_Outputs;
        };

        for (const Gene& gene : m_Genes)
        {
            if (gene.IsEnabled() && gene.GetWeight() != 0.0f)
            {
                int geneIn = translateIndex(gene.GetIn());
                int geneOut = translateIndex(gene.GetOut());

                neurons[geneOut].links.emplace_back(geneIn, gene.GetWeight());
            }
        }

        return neuralNetwork.Make(m_Inputs, m_Outputs, std::move(neurons));
    }

    void UpdateGlobalRank(int globalRank) { m_GlobalRank = globalRank; }
    void SetScore(float score) { m_Score = score; }

    static int GetInnovation() { return ms_Innovation; }
    static int GetNewInnovation() { return ms_Innovation++; }
    static void ResetInnovation() { ms_Innovation = 0; }

    const std::vector<Gene>& GetGenes() const { return m_Genes; }
    const std::unordered_map<Mutations, float>& GetMutationChances() const { return m_MutationChances; }
    int GetInputs() const { return m_Inputs; }
    int GetOutputs() const { return m_Outputs; }
    int GetMaxNeurons() const { return m_MaxNeurons; }
    int GetGlobalRank() const { return m_GlobalRank; }
    float GetScore() const { return m_Score; }

private:
    void PointMutate()
    {
        float step = m_MutationChances[Mutations::Step];
        for (Gene& gene : m_Genes)
        {
            if (BrainFramework::RandomFloat() < k_PerturbChance)
            {
                gene.SetWeight(gene.GetWeight() + BrainFramework::RandomFloat() * step * 2.0f - step);
            }
            else
            {
                gene.SetWeight(BrainFramework::RandomFloat() * 4.0f - 2.0f);
            }
        }
    }

    void LinkMutate(bool forceInputBias)
    {
        int neuron1 = 0;
        if (forceInputBias)
        {
            // Inputs
            neuron1 = BrainFramework::RandomInt(0, m_Inputs - 1);
        }
        else
        {
            // Any, except outputs, with the same proba
            neuron1 = BrainFramework::RandomInt(0, m_MaxNeurons - m_Outputs - 1);
            if (neuron1 >= m_Inputs)
                neuron1 += m_Outputs;
        }

        int neuron2 = 0;
        // We want any outputs or greater than neuron1 but not an input
        if (neuron1 < m_Inputs)
        {
            // In this case, any neurons not an input
            neuron2 = BrainFramework::RandomInt(m_Inputs, m_MaxNeurons - 1);
        }
        else
        {
            // In this case, "add" the outputs at the end
            neuron2 = BrainFramework::RandomInt(neuron1 + 1, m_MaxNeurons + m_Outputs - 1);
            if (neuron2 >= m_MaxNeurons)
            {
                neuron2 -= m_MaxNeurons;
                neuron2 += m_Inputs;
            }
        }

        // Already existing skip
        for (const Gene& gene : m_Genes)
            if (gene.GetIn() == neuron1 && gene.GetOut() == neuron2)
                return;

        assert(neuron1 < m_MaxNeurons);
        assert(neuron2 < m_MaxNeurons);
        assert(neuron1 >= 0);
        assert(neuron2 >= m_Inputs);

        m_Genes.emplace_back(neuron1, neuron2, BrainFramework::RandomFloat() * 4.0f - 2.0f, true, GetNewInnovation());
    }

    void NodeMutate()
    {
        if (m_Genes.empty())
            return;

        m_MaxNeurons++;

        const int initialGeneIndex = BrainFramework::RandomIndex(m_Genes);
        {
            Gene& initialGene = m_Genes[initialGeneIndex];
            if (initialGene.IsEnabled())
                return;
            initialGene.SetEnabled(false);
        }

        Gene& gene1 = m_Genes.emplace_back();
        gene1.CopyFrom(m_Genes[initialGeneIndex]); // Emplace_back might break previous reference
        gene1.SetOut(m_MaxNeurons - 1);
        gene1.SetWeight(1.0f);
        gene1.SetInnovation(GetNewInnovation());
        gene1.SetEnabled(true);

        Gene& gene2 = m_Genes.emplace_back();
        gene2.CopyFrom(m_Genes[initialGeneIndex]); // Emplace_back might break previous reference
        gene2.SetIn(m_MaxNeurons - 1);
        gene2.SetInnovation(GetNewInnovation());
        gene2.SetEnabled(true);
    }

    void EnableDisableMutate(bool enable)
    {
        std::vector<Gene*> candidates;
        candidates.reserve(m_Genes.size());
        for (Gene& gene : m_Genes)
            if (gene.IsEnabled() != enable)
                candidates.push_back(&gene);

        if (candidates.empty())
            return;

        Gene* candidate = candidates[BrainFramework::RandomIndex(candidates)];
        candidate->SetEnabled(!candidate->IsEnabled());
    }

private:
    std::vector<Gene> m_Genes;
    std::unordered_map<Mutations, float> m_MutationChances;
    int m_Inputs{ 0 };
    int m_Outputs{ 0 };
    int m_MaxNeurons{ 0 };
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

    static inline int ms_Innovation = 0;
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

    void Update()
    {
        SortGenomes();

        const float previousScore = m_TopScore;
        m_TopScore = m_Genomes[0].GetScore();

        if (m_TopScore > previousScore)
        {
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

        const std::size_t remaining = cutToOne ? 1 : static_cast<std::size_t>(std::ceil(m_Genomes.size() / 2.0f));
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

class NEATModel : public BrainFramework::Model
{
public:
    NEATModel() = default;
    NEATModel(const NEATModel&) = delete;
    NEATModel& operator=(const NEATModel&) = delete;

    const char* GetName() const override { return "NEAT"; }

    void DisplayImGui() override
    {
        ImGui::Text("MaxScore: %f", GetMaxScore());
        ImGui::Text("Generation: %d", GetGeneration());
        ImGui::Text("Species: %d", static_cast<int>(GetSpecies().size()));

        int genomes = 0;
        for (const Species& species : GetSpecies())
        {
            genomes += static_cast<int>(species.GetGenomes().size());
        }
        ImGui::Text("Genomes: %d", genomes);

        ImGui::Text("Innovations: %d", Genome::GetInnovation());
    }

    bool StartTraining(const BrainFramework::Simulation& simulation) override
    {
        if (m_Species.empty())
        {
            Reset();

            for (int i = 0; i < k_Population; ++i)
            {
                Genome genome;
                genome.Initialize(simulation.GetInputsCount(), simulation.GetOutputsCount());
                genome.Mutate();
                AddToSpecies(genome);
            }
        }

        m_CurrentSpecies = 0;
        m_CurrentGenome = 0;
        return Model::StartTraining(simulation);
    }

    bool Train(BrainFramework::Simulation& simulation) override
    {
        Genome& genome = m_Species[m_CurrentSpecies].GetGenomes()[m_CurrentGenome];

        BrainFramework::NeuralNetwork neuralNetwork;
        if (!genome.MakeNeuralNetwork(neuralNetwork))
        {
            return false;
        }

        if (simulation.GetResult() != BrainFramework::Simulation::Result::Initialized)
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

        return true;
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

    void Reset()
    {
        Genome::ResetInnovation();
        m_Species.clear();
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

    void AddToSpecies(const Genome& genome)
    {
        for (Species& species : m_Species)
        {
            if (species.SameSpecies(genome))
            {
                species.GetGenomes().push_back(genome);
                return;
            }
        }
        
        Species& species = m_Species.emplace_back();
        species.GetGenomes().push_back(genome);
    }

    void NewGeneration()
    {
        m_MaxScore = k_ResetMaxScore;
        for (Species& species : m_Species)
        {
            species.Update();
            if (species.GetTopScore() > m_MaxScore)
                m_MaxScore = species.GetTopScore();
        }

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
            if (breedCount <= 0 && m_Species.size() > 1)
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
            AddToSpecies(genome);
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

        // Ranks are inverted to be used for Fitness computation
        const int allGenomesCount = static_cast<int>(allGenomes.size());
        for (int i = 0; i < allGenomesCount; ++i)
        {
            allGenomes[i]->UpdateGlobalRank(i + 1);
        }
    }

    float GetMaxScore() const { return m_MaxScore; }
    int GetGeneration() const { return m_Generation; }
    const std::vector<Species>& GetSpecies() const { return m_Species; }

    static constexpr int k_Population = 300;
    static constexpr float k_ResetMaxScore = -100000.0f;

private:
    Genome m_BestGenome;
    std::vector<Species> m_Species;
    float m_MaxScore{ k_ResetMaxScore };
    int m_Generation{ 0 };
    int m_CurrentSpecies{ 0 };
    int m_CurrentGenome{ 0 };
};

} // namespace NEAT
