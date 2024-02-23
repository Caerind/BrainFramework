#pragma once

#include "../src/BrainFramework.hpp"

namespace NEETL
{

    class Genome
    {
    public:
        enum class Mutations
        {
            AddNeuron,
            RemoveNeuron,
            AddLayer,
            AlterateWeights,
            Step,

            COUNT
        };

        static constexpr float k_PerturbChance = 0.90f; 
        static constexpr float k_AddNeuronChance = 0.15f;
        static constexpr float k_RemoveNeuronChance = 0.1f;
        static constexpr float k_AddLayerChance = 0.01f;
        static constexpr float k_AlterateWeightsChance = 1.1f;
        static constexpr float k_StepSize = 0.1f;
        static constexpr int k_InitialIntermediateLayers = 3;

        Genome()
        {
            m_MutationChances.reserve(static_cast<std::size_t>(Mutations::COUNT));
            m_MutationChances[Mutations::AddNeuron] = k_AddNeuronChance;
            m_MutationChances[Mutations::RemoveNeuron] = k_RemoveNeuronChance;
            m_MutationChances[Mutations::AddLayer] = k_AddLayerChance;
            m_MutationChances[Mutations::AlterateWeights] = k_AlterateWeightsChance;
            m_MutationChances[Mutations::Step] = k_StepSize;
        }

        //Genome(const Genome&) = delete;
        //Genome& operator=(const Genome&) = delete;

        void CopyFrom(const Genome& other)
        {
            m_Inputs = other.m_Inputs;
            m_Outputs = other.m_Outputs;

            m_LayerSizes = other.m_LayerSizes;
            m_Weights = other.m_Weights;
            assert(m_LayerSizes.size() >= 2);
            assert(m_LayerSizes[0] == m_Inputs);
            assert(m_LayerSizes.back() == m_Outputs);

            m_MutationChances[Mutations::AddNeuron] = other.m_MutationChances.at(Mutations::AddNeuron);
            m_MutationChances[Mutations::RemoveNeuron] = other.m_MutationChances.at(Mutations::RemoveNeuron);
            m_MutationChances[Mutations::AddLayer] = other.m_MutationChances.at(Mutations::AddLayer);
            m_MutationChances[Mutations::AlterateWeights] = other.m_MutationChances.at(Mutations::AlterateWeights);
            m_MutationChances[Mutations::Step] = other.m_MutationChances.at(Mutations::Step);
        }

        void Crossover(const Genome& genome1, const Genome& genome2)
        {
            // Make sure genome1 is the higher fitness genome
            if (genome2.m_AverageScore > genome1.m_AverageScore)
            {
                Crossover(genome2, genome1);
                return;
            }

            assert(genome1.m_Inputs == genome2.m_Inputs);
            assert(genome1.m_Outputs == genome2.m_Outputs);

            m_Inputs = genome1.m_Inputs;
            m_Outputs = genome1.m_Outputs;

            m_LayerSizes = genome1.m_LayerSizes;
            assert(m_LayerSizes.size() >= 2);
            assert(m_LayerSizes[0] == m_Inputs);
            assert(m_LayerSizes.back() == m_Outputs);

            // TODO : Mix with genome2 ?
            m_Weights = genome1.m_Weights;

            for (int i = 0; i < static_cast<int>(Mutations::COUNT); ++i)
            {
                const Mutations mut = static_cast<Mutations>(i);
                m_MutationChances[mut] = (BrainFramework::RandomBool()) ? genome1.m_MutationChances.at(mut) : genome2.m_MutationChances.at(mut);
            }
        }

        void Initialize(int inputs, int outputs)
        {
            m_Inputs = inputs;
            m_Outputs = outputs;

            m_LayerSizes.resize(2);
            m_LayerSizes[0] = inputs;
            m_LayerSizes[1] = outputs;

            m_Weights.resize(inputs * outputs);
            int index = 0;
            for (int i = 0; i < m_Inputs; ++i)
            {
                for (int o = 0; o < m_Outputs; ++o, ++index)
                {
                    m_Weights[index] = BrainFramework::RandomFloat() * 4.0f - 2.0f;
                }
            }

            for (int j = 0; j < k_InitialIntermediateLayers; ++j)
            {
                BrainFramework::LayeredNeuralNetwork::AddLayer(m_LayerSizes, m_Weights, 1);
            }
        }

        void Mutate()
        {
            // Alterate mutation chances
            for (auto& mutationChance : m_MutationChances)
            {
                mutationChance.second *= (BrainFramework::RandomBool()) ? 0.95f : 1.05263f;
            }

            float p = static_cast<float>(m_MutationChances[Mutations::AlterateWeights]);
            while (p > 0.0f)
            {
                if (BrainFramework::RandomFloat() < p)
                {
                    // Alterate Weights
                    float step = m_MutationChances[Mutations::Step];
                    for (float& v : m_Weights)
                    {
                        if (BrainFramework::RandomFloat() < k_PerturbChance)
                        {
                            v += (BrainFramework::RandomFloat() * step * 2.0f - step);
                        }
                        else
                        {
                            v = BrainFramework::RandomFloat() * 4.0f - 2.0f;
                        }
                    }
                }
                p -= 1.0f;
            }

            int intermediateLayerCount = static_cast<int>(m_LayerSizes.size() - 2);

            p = static_cast<float>(m_MutationChances[Mutations::AddNeuron]);
            while (p > 0.0f && intermediateLayerCount > 0)
            {
                if (BrainFramework::RandomFloat() < p)
                {
                    // Add Neuron
                    int layerIndex = 1 + BrainFramework::RandomInt(0, intermediateLayerCount);
                    BrainFramework::LayeredNeuralNetwork::AddNeuronOnLayer(m_LayerSizes, m_Weights, layerIndex);
                }
                p -= 1.0f;
            }

            p = static_cast<float>(m_MutationChances[Mutations::RemoveNeuron]);
            while (p > 0.0f && intermediateLayerCount > 0)
            {
                if (BrainFramework::RandomFloat() < p)
                {
                    // Remove Neuron
                    int layerIndex = 1 + BrainFramework::RandomInt(0, intermediateLayerCount);
                    BrainFramework::LayeredNeuralNetwork::RemoveNeuronOnLayer(m_LayerSizes, m_Weights, layerIndex);
                }
                p -= 1.0f;
            }

            p = static_cast<float>(m_MutationChances[Mutations::AddLayer]);
            while (p > 0.0f)
            {
                if (BrainFramework::RandomFloat() < p)
                {
                    // Add Layer
                    const int newLayerIndex = BrainFramework::RandomInt(1, static_cast<int>(m_LayerSizes.size() - 1));
                    BrainFramework::LayeredNeuralNetwork::AddLayer(m_LayerSizes, m_Weights, newLayerIndex);
                }
                p -= 1.0f;
            }
        }

        bool MakeNeuralNetwork(BrainFramework::LayeredNeuralNetwork& neuralNetwork) const
        {
            return neuralNetwork.Make(m_LayerSizes, m_Weights);
        }

        void EndBatch(float score)
        {
            m_Score = score;
            m_Lifetime++;
            m_AverageScore = ((m_Lifetime - 1) * m_AverageScore + m_Score) / m_Lifetime;
        }

        const std::unordered_map<Mutations, float>& GetMutationChances() const { return m_MutationChances; }
        int GetInputs() const { return m_Inputs; }
        int GetOutputs() const { return m_Outputs; }
        float GetScore() const { return m_Score; }
        float GetAverageScore() const { return m_AverageScore; }
        int GetLifetime() const { return m_Lifetime; }

        int GetNeuronsCount() const
        {
            int count = 0;
            for (int v : m_LayerSizes)
                count += v;
            return count;
        }

        int GetLinksCount() const
        {
            int count = 0;
            const int layers = static_cast<int>(m_LayerSizes.size());
            for (int i = 1; i < layers; ++i)
            {
                count += (m_LayerSizes[i - 1] * m_LayerSizes[i]);
            }
            return count;
        }

        int GetLayersCount() const { return static_cast<int>(m_LayerSizes.size()); }

    private:
        std::vector<int> m_LayerSizes;
        std::vector<float> m_Weights;
        std::unordered_map<Mutations, float> m_MutationChances;
        int m_Inputs{ 0 };
        int m_Outputs{ 0 };
        float m_Score{ 0.0f };
        float m_AverageScore{ 0.0f };
        int m_Lifetime{ 0 };

        static inline int ms_Innovation = 0;
    };

    class NEETLModel : public BrainFramework::Model
    {
    public:
        NEETLModel() = default;
        NEETLModel(const NEETLModel&) = delete;
        NEETLModel& operator=(const NEETLModel&) = delete;

        const char* GetName() const override { return "NEETLModel"; }

        void DisplayImGui() override
        {
            ImGui::Text("Generation: %d", GetGeneration());
            ImGui::Text("MaxLifetime: %d", m_MaxLifetime);
            ImGui::Text("AverageScoreTop5: %f", m_AverageScoreTop5);
            ImGui::Text("AverageScoreTop10: %f", m_AverageScoreTop10);
            ImGui::Text("AverageScore: %f", m_AverageScore);

            ImGui::PlotHistogram("MaxLifetime", m_LifetimeArray.data(), static_cast<int>(m_LifetimeArray.size()), 0, NULL, FLT_MAX, FLT_MAX, ImVec2(0, 80));
            ImGui::PlotHistogram("AverageScoreTop5", m_AverageScoreTop5Array.data(), static_cast<int>(m_AverageScoreTop5Array.size()), 0, NULL, FLT_MAX, FLT_MAX, ImVec2(0, 80));
            ImGui::PlotHistogram("AverageScoreTop10", m_AverageScoreTop10Array.data(), static_cast<int>(m_AverageScoreTop10Array.size()), 0, NULL, FLT_MAX, FLT_MAX, ImVec2(0, 80));
            ImGui::PlotHistogram("AverageScore", m_AverageScoreArray.data(), static_cast<int>(m_AverageScoreArray.size()), 0, NULL, FLT_MAX, FLT_MAX, ImVec2(0, 80));

            if (!m_Bests.empty())
            {
                ImGui::Text("BestGenome:");
                ImGui::Indent();
                ImGui::Text("Neurons: %d", m_Bests[0].GetNeuronsCount());
                ImGui::Text("Links: %d", m_Bests[0].GetLinksCount());
                ImGui::Unindent();
            }
        }

        bool PrepareTraining(const BrainFramework::ISimulation& simulation) override
        {
            if (m_Genomes.empty())
            {
                for (int i = 0; i < k_Population; ++i)
                {
                    Genome& genome = m_Genomes.emplace_back();
                    genome.Initialize(simulation.GetRLInputsCount(), simulation.GetRLOutputsCount());
                    genome.Mutate();
                }
            }

            m_Bests.clear();
            m_Bests.resize(k_BestCount);

            m_CurrentGenome = 0;
            m_CurrentGenomeEvaluation = 0;
            m_CurrentGenomeScoreSum = 0.0f;

            return true;
        }

        bool StartEvaluation(std::unique_ptr<BrainFramework::NeuralNetwork>& neuralNetwork) override
        {
            Genome& genome = m_Genomes[m_CurrentGenome];

            std::unique_ptr<BrainFramework::LayeredNeuralNetwork> layeredNeuralNetwork = std::make_unique<BrainFramework::LayeredNeuralNetwork>();
            if (!genome.MakeNeuralNetwork(*layeredNeuralNetwork))
            {
                return false;
            }

            neuralNetwork = std::move(layeredNeuralNetwork);

            return true;
        }

        bool EndEvalutation(float result) override
        {
            Genome& genome = m_Genomes[m_CurrentGenome];

            m_CurrentGenomeEvaluation++;
            m_CurrentGenomeScoreSum += result;

            constexpr int batchSize = 10;
            if (m_CurrentGenomeEvaluation >= batchSize)
            {
                genome.EndBatch(m_CurrentGenomeScoreSum / batchSize);

                m_CurrentGenomeEvaluation = 0;
                m_CurrentGenomeScoreSum = 0.0f;

                NextGenome();
            }

            return true;
        }

        bool MakeBestNeuralNetwork(std::unique_ptr<BrainFramework::NeuralNetwork>& neuralNetwork, int index = 0) override
        {
            index = index % k_BestCount;

            std::unique_ptr<BrainFramework::LayeredNeuralNetwork> layeredNeuralNetwork = std::make_unique<BrainFramework::LayeredNeuralNetwork>();
            const bool result = m_Bests[index].MakeNeuralNetwork(*layeredNeuralNetwork);

            neuralNetwork = std::move(layeredNeuralNetwork);

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
            for (Genome& genome : m_Genomes)
            {
                if (genome.GetLifetime() > m_MaxLifetime) m_MaxLifetime = genome.GetLifetime();
            }

            std::sort(m_Genomes.begin(), m_Genomes.end(), [&](const Genome& a, const Genome& b)
                {
                    return a.GetAverageScore() > b.GetAverageScore();
                });

            // Save our best
            for (int i = 0; i < k_BestCount; ++i)
            {
                m_Bests[i].CopyFrom(m_Genomes[0]);
            }

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
        static constexpr int k_BestCount = 10;

        static constexpr int k_HistogramValues = 300;

        static constexpr float k_ResetMaxScore = -100000.0f;

    private:
        std::vector<Genome> m_Bests;
        std::vector<Genome> m_Genomes;
        int m_CurrentGenome{ 0 };
        int m_CurrentGenomeEvaluation{ 0 };
        float m_CurrentGenomeScoreSum{ 0 };

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

} // namespace NEETL
