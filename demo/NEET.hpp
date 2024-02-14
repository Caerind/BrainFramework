#pragma once

#include "../src/BrainFramework.hpp"

namespace NEET
{

class Gene
{
public:
    Gene() = default;
    Gene(int in, int out, float weight, bool enabled)
        : m_Weight(weight)
        , m_In(in)
        , m_Out(out)
        , m_Enabled(enabled)
    {
    }

    void CopyFrom(const Gene& other)
    {
        m_Weight = other.m_Weight;
        m_In = other.m_In;
        m_Out = other.m_Out;
        m_Enabled = other.m_Enabled;
    }

    void SetIn(int in) { m_In = in; }
    void SetOut(int out) { m_Out = out; }
    void SetWeight(float weight) { m_Weight = weight; }
    void SetEnabled(bool enabled) { m_Enabled = enabled; }

    int GetIn() const { return m_In; }
    int GetOut() const { return m_Out; }
    float GetWeight() const { return m_Weight; }
    bool IsEnabled() const { return m_Enabled; }

private:
    float m_Weight{ 0.0f };
    int m_In{ 0 };
    int m_Out{ 0 };
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
        Step,

        COUNT
    };

    Genome()
    {
        m_MutationChances.reserve(static_cast<std::size_t>(Mutations::COUNT));
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
        // Make sure genome1 is the higher genome
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

        const int sizeMax = static_cast<int>((genome1.m_Genes.size() > genome2.m_Genes.size()) ? genome1.m_Genes.size() : genome2.m_Genes.size());

        m_Genes.clear();
        m_Genes.reserve(sizeMax);
        for (int i = 0; i < sizeMax; ++i)
        {
            bool can1 = i < static_cast<int>(genome1.m_Genes.size());
            bool can2 = i < static_cast<int>(genome2.m_Genes.size());

            if (can1 && can2)
            {
                can1 = BrainFramework::RandomBool();
                can2 = !can1;
            }

            Gene& gene = m_Genes.emplace_back();
            if (can1)
            {
                gene.CopyFrom(genome1.m_Genes[i]);
            }
            else if (can2)
            {
                gene.CopyFrom(genome2.m_Genes[i]);
            }
            else
            {
                assert(false);
            }
        }

        m_MutationChances.reserve(static_cast<std::size_t>(Mutations::COUNT));

        for (int i = 0; i < static_cast<int>(Mutations::COUNT); ++i)
        {
            const Mutations mut = static_cast<Mutations>(i);
            m_MutationChances[mut] = BrainFramework::RandomBool() ? genome1.m_MutationChances.at(mut) : genome2.m_MutationChances.at(mut);
        }
    }

    void Initialize(int inputs, int outputs)
    {
        m_Inputs = inputs;
        m_Outputs = outputs;
        m_MaxNeurons = m_Inputs + m_Outputs;

        /*
        m_Genes.reserve(m_Inputs * m_Outputs);
        for (int o = 0; o < m_Outputs; ++o)
            for (int i = 0; i < m_Inputs; ++i)
                if (BrainFramework::RandomBool())
                    m_Genes.emplace_back(i, o + m_Inputs, BrainFramework::RandomFloat() * 4.0f - 2.0f, true);
        */
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

    bool MakeNeuralNetwork(BrainFramework::BasicNeuralNetwork& neuralNetwork) const
    {
        std::vector<BrainFramework::BasicNeuralNetwork::Neuron> neurons;
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

    void EndBatch(float score)
    {
        m_Score = score;
        m_Lifetime++;
        m_AverageScore = ((m_Lifetime - 1) * m_AverageScore + score) / m_Lifetime;
    }

    const std::vector<Gene>& GetGenes() const { return m_Genes; }
    const std::unordered_map<Mutations, float>& GetMutationChances() const { return m_MutationChances; }
    int GetInputs() const { return m_Inputs; }
    int GetOutputs() const { return m_Outputs; }
    int GetMaxNeurons() const { return m_MaxNeurons; }
    float GetScore() const { return m_Score; }
    float GetAverageScore() const { return m_AverageScore; }
    int GetLifetime() const { return m_Lifetime; }

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

        m_Genes.emplace_back(neuron1, neuron2, BrainFramework::RandomFloat() * 4.0f - 2.0f, true);
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
        gene1.SetEnabled(true);

        Gene& gene2 = m_Genes.emplace_back();
        gene2.CopyFrom(m_Genes[initialGeneIndex]); // Emplace_back might break previous reference
        gene2.SetIn(m_MaxNeurons - 1);
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
    float m_Score{ 0.0f };
    float m_AverageScore{ 0.0f };
    int m_Lifetime{ 0 };

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

class NEETModel : public BrainFramework::Model
{
public:
    NEETModel() = default;
    NEETModel(const NEETModel&) = delete;
    NEETModel& operator=(const NEETModel&) = delete;

    const char* GetName() const override { return "NEET"; }

    void DisplayImGui() override
    {
        ImGui::Text("Generation: %d", GetGeneration());
        ImGui::Text("maxLifetime: %d", m_MaxLifetime);
        ImGui::Text("averageScoreTop5: %f", m_AverageScoreTop5);
        ImGui::Text("averageScoreTop10: %f", m_AverageScoreTop10);
        ImGui::Text("averageScore: %f", m_AverageScore);

        ImGui::PlotHistogram("MaxLifetime", m_LifetimeArray.data(), static_cast<int>(m_LifetimeArray.size()), 0, NULL, FLT_MAX, FLT_MAX, ImVec2(0, 80));
        ImGui::PlotHistogram("AverageScoreTop5", m_AverageScoreTop5Array.data(), static_cast<int>(m_AverageScoreTop5Array.size()), 0, NULL, FLT_MAX, FLT_MAX, ImVec2(0, 80));
        ImGui::PlotHistogram("AverageScoreTop10", m_AverageScoreTop10Array.data(), static_cast<int>(m_AverageScoreTop10Array.size()), 0, NULL, FLT_MAX, FLT_MAX, ImVec2(0, 80));
        ImGui::PlotHistogram("AverageScore", m_AverageScoreArray.data(), static_cast<int>(m_AverageScoreArray.size()), 0, NULL, FLT_MAX, FLT_MAX, ImVec2(0, 80));

        ImGui::Text("BestGenome:");
        ImGui::Indent();
        ImGui::Text("Neurons: %d", m_BestGenome.GetMaxNeurons());
        int genes = 0;
        for (const Gene& gene : m_BestGenome.GetGenes())
            if (gene.IsEnabled() && gene.GetWeight() != 0.0f)
                genes++;
        ImGui::Text("Genes: %d", genes);
        ImGui::Unindent();
    }

    bool StartTraining(const BrainFramework::Simulation& simulation) override
    {
        if (m_Genomes.empty())
        {
            for (int i = 0; i < k_Population; ++i)
            {
                Genome& genome = m_Genomes.emplace_back();
                genome.Initialize(simulation.GetInputsCount(), simulation.GetOutputsCount());
                genome.Mutate();
            }
        }

        m_CurrentGenome = 0;
        return Model::StartTraining(simulation);
    }

    bool Train(BrainFramework::Simulation& simulation) override
    {
        Genome& genome = m_Genomes[m_CurrentGenome];

        BrainFramework::BasicNeuralNetwork neuralNetwork;
        if (!genome.MakeNeuralNetwork(neuralNetwork))
        {
            return false;
        }

        constexpr int batchSize = 10;
        float scoreSum = 0.0f;

        for (int i = 0; i < batchSize; ++i)
        {
            if (simulation.GetResult() != BrainFramework::Simulation::Result::Initialized)
            {
                simulation.Initialize(neuralNetwork);
            }

            auto result = BrainFramework::Simulation::Result::Initialized;
            do
            {
                result = simulation.Step(neuralNetwork);
            } while (result == BrainFramework::Simulation::Result::Ongoing);

            scoreSum += simulation.GetScore();
        }

        genome.EndBatch(scoreSum / batchSize);

        NextGenome();        

        return true;
    }

    bool StopTraining(const BrainFramework::Simulation& simulation) override
    {
        Model::StopTraining(simulation);
        return true;
    }

    bool MakeBestNeuralNetwork(std::unique_ptr<BrainFramework::NeuralNetwork>& neuralNetwork) override
    {
        if (IsTraining())
            return false;

        std::unique_ptr<BrainFramework::BasicNeuralNetwork> basicNeuralNetwork = std::make_unique<BrainFramework::BasicNeuralNetwork>();
        const bool result = m_BestGenome.MakeNeuralNetwork(*basicNeuralNetwork);

        neuralNetwork = std::move(basicNeuralNetwork);

        return result;
    }

    void NextGenome()
    {
        m_CurrentGenome++;
        if (m_CurrentGenome >= static_cast<int>(m_Genomes.size()))
        {
            m_CurrentGenome = 0;
            NewGeneration();
        }
    }

    void NewGeneration()
    {
        m_MaxLifetime = 0;
        m_AverageScore = 0.0f;
        for (Genome& genome : m_Genomes)
        {
            if (genome.GetLifetime() > m_MaxLifetime) m_MaxLifetime = genome.GetLifetime();
            m_AverageScore += genome.GetAverageScore();
        }

        std::sort(m_Genomes.begin(), m_Genomes.end(), [&](const Genome& a, const Genome& b)
        {
            return a.GetAverageScore() > b.GetAverageScore();
        });

        // Save our best
        m_BestGenome.CopyFrom(m_Genomes[0]);

        // Cull the bottom
        m_Genomes.erase(m_Genomes.begin() + m_Genomes.size() / k_Cut, m_Genomes.end());

        const int size = static_cast<int>(m_Genomes.size());

        // Compute health
        m_AverageScore = 0.0f; // Recompute with removed ones
        m_AverageScoreTop5 = 0.0f;
        m_AverageScoreTop10 = 0.0f;
        for (int i = 0; i < size; ++i)
        {
            const float avg = m_Genomes[i].GetAverageScore();
            if (i < 5) m_AverageScoreTop5 += avg;
            if (i < 10) m_AverageScoreTop10 += avg;
            m_AverageScore += avg;
        }
        m_AverageScore /= size;
        m_AverageScoreTop5 /= 5;
        m_AverageScoreTop10 /= 10;

        // Add to histograms
        m_LifetimeArray.push_back(static_cast<float>(m_MaxLifetime));
        if (m_LifetimeArray.size() > k_HistogramValues) m_LifetimeArray.erase(m_LifetimeArray.begin());
        m_AverageScoreTop5Array.push_back(m_AverageScoreTop5);
        if (m_AverageScoreTop5Array.size() > k_HistogramValues) m_AverageScoreTop5Array.erase(m_AverageScoreTop5Array.begin());
        m_AverageScoreTop10Array.push_back(m_AverageScoreTop10);
        if (m_AverageScoreTop10Array.size() > k_HistogramValues) m_AverageScoreTop10Array.erase(m_AverageScoreTop10Array.begin());
        m_AverageScoreArray.push_back(m_AverageScore);
        if (m_AverageScoreArray.size() > k_HistogramValues) m_AverageScoreArray.erase(m_AverageScoreArray.begin());

        // Breed from any adults
        for (int i = size; i < k_Population; ++i)
        {
            int parent1Index = BrainFramework::RandomInt(0, size / 2);
            int parent2Index = BrainFramework::RandomInt(0, size / 2);

            Genome& child = m_Genomes.emplace_back();

            constexpr float k_CrossoverChance = 0.75f;
            if (BrainFramework::RandomFloat() < k_CrossoverChance && parent1Index != parent2Index)
            {
                child.Crossover(m_Genomes[parent1Index], m_Genomes[parent2Index]);
            }
            else
            {
                child.CopyFrom(m_Genomes[parent1Index]);
            }

            child.Mutate();
        }

        m_Generation++;
    }

    int GetGeneration() const { return m_Generation; }
    const std::vector<Genome>& GetGenomes() const { return m_Genomes; }

    static constexpr int k_Population = 300;
    static constexpr int k_Cut = 3;

    static constexpr int k_HistogramValues = 300;

    static constexpr float k_ResetMaxScore = -100000.0f;

private:
    Genome m_BestGenome;
    std::vector<Genome> m_Genomes;
    int m_CurrentGenome{ 0 };

    int m_Generation{ 0 };
    int m_MaxLifetime{ 0 };
    float m_AverageScoreTop5{ 0.0f };
    float m_AverageScoreTop10{ 0.0f };
    float m_AverageScore{ 0.0f };
    std::vector<float> m_LifetimeArray;
    std::vector<float> m_AverageScoreTop5Array;
    std::vector<float> m_AverageScoreTop10Array;
    std::vector<float> m_AverageScoreArray;
};

} // namespace NEET
