#pragma once

#include "../src/BrainFramework.hpp"

namespace NEET
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
        static constexpr float k_AddNeuronChance = 0.4f;
        static constexpr float k_RemoveNeuronChance = 0.3f;
        static constexpr float k_AddLayerChance = 0.01f;
        static constexpr float k_AlterateWeightsChance = 1.1f;
        static constexpr float k_StepSize = 0.1f;

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

            m_LayerSizes.reserve(2);
            m_LayerSizes[0] = inputs;
            m_LayerSizes[1] = outputs;

            m_Weights.reserve(inputs * outputs);
            int index = 0;
            for (int i = 0; i < m_Inputs; ++i)
            {
                for (int o = 0; o < m_Outputs; ++o, ++index)
                {
                    m_Weights[index] = BrainFramework::RandomFloat() * 4.0f - 2.0f;
                }
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
                    int layerIndex = BrainFramework::RandomInt(0, intermediateLayerCount) + 1;
                    const int oldLayerSize = m_LayerSizes[layerIndex];
                    m_LayerSizes[layerIndex]++;

                    int weightBeginIndex = 0;
                    for (int i = 1; i < layerIndex; ++i)
                        weightBeginIndex += (m_LayerSizes[i] * m_LayerSizes[i - 1]);

                    // Add links from previous layer
                    const int previousLayerNeurons = m_LayerSizes[layerIndex - 1];
                    for (int i = 0; i < previousLayerNeurons; ++i)
                    {
                        const int index = weightBeginIndex + oldLayerSize * (i + 1) - 1;
                        m_Weights.insert(m_Weights.begin() + index, BrainFramework::RandomFloat() * 4.0f - 2.0f);
                    }

                    weightBeginIndex += previousLayerNeurons * m_LayerSizes[layerIndex];

                    // Add links for next layer
                    const int nextLayerNeurons = m_LayerSizes[layerIndex + 1];
                    const int nextWeightsIndex = weightBeginIndex + oldLayerSize * nextLayerNeurons;
                    m_Weights.insert(m_Weights.begin() + nextWeightsIndex, nextLayerNeurons, 0.0f);
                    for (int i = 0; i < nextLayerNeurons; ++i)
                    {
                        m_Weights[nextWeightsIndex + i] = BrainFramework::RandomFloat() * 4.0f - 2.0f;
                    }
                }
                p -= 1.0f;
            }

            p = static_cast<float>(m_MutationChances[Mutations::RemoveNeuron]);
            while (p > 0.0f && intermediateLayerCount > 0)
            {
                if (BrainFramework::RandomFloat() < p)
                {
                    // Remove Neuron
                    int layerIndex = BrainFramework::RandomInt(0, intermediateLayerCount) + 1;
                    const int oldLayerSize = m_LayerSizes[layerIndex];
                    if (oldLayerSize <= 1)
                        continue;
                    m_LayerSizes[layerIndex]--;

                    int weightBeginIndex = 0;
                    for (int i = 1; i < layerIndex; ++i)
                        weightBeginIndex += (m_LayerSizes[i] * m_LayerSizes[i - 1]);
                    const int previousLayerNeurons = m_LayerSizes[layerIndex - 1];

                    weightBeginIndex += oldLayerSize * previousLayerNeurons;

                    // Remove links from next layer (do this one first to optimize remove operations on std::vector)
                    const int nextLayerNeurons = m_LayerSizes[layerIndex + 1];
                    const auto nextLayerBegin = m_Weights.begin() + weightBeginIndex + m_LayerSizes[layerIndex] * nextLayerNeurons;
                    const auto nextLayerLast = nextLayerBegin + nextLayerNeurons;
                    m_Weights.erase(nextLayerBegin, nextLayerLast);

                    weightBeginIndex -= oldLayerSize * previousLayerNeurons;

                    // Remove links from previous layer
                    // TODO : Reverse order to optimize remove operations a bit
                    const int previousLayerNeurons = m_LayerSizes[layerIndex - 1];
                    for (int i = 0; i < previousLayerNeurons; ++i)
                    {
                        const int index = weightBeginIndex + oldLayerSize * (i + 1) - 1 - i;
                        m_Weights.erase(m_Weights.begin() + index);
                    }
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
                    const int previousLayerSize = m_LayerSizes[newLayerIndex - 1];
                    const int nextLayerSize = m_LayerSizes[newLayerIndex];
                    const int newLayerSize = BrainFramework::RandomInt(std::min(previousLayerSize, nextLayerSize), std::max(previousLayerSize, nextLayerSize));
                    m_LayerSizes.insert(m_LayerSizes.begin() + newLayerIndex, newLayerSize);
                    // TODO
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
            ImGui::Text("MaxLifetime: %d", m_MaxLifetime);
            ImGui::Text("AverageScoreTop5: %f", m_AverageScoreTop5);
            ImGui::Text("AverageScoreTop10: %f", m_AverageScoreTop10);
            ImGui::Text("AverageScore: %f", m_AverageScore);

            ImGui::PlotHistogram("MaxLifetime", m_LifetimeArray.data(), static_cast<int>(m_LifetimeArray.size()), 0, NULL, FLT_MAX, FLT_MAX, ImVec2(0, 80));
            ImGui::PlotHistogram("AverageScoreTop5", m_AverageScoreTop5Array.data(), static_cast<int>(m_AverageScoreTop5Array.size()), 0, NULL, FLT_MAX, FLT_MAX, ImVec2(0, 80));
            ImGui::PlotHistogram("AverageScoreTop10", m_AverageScoreTop10Array.data(), static_cast<int>(m_AverageScoreTop10Array.size()), 0, NULL, FLT_MAX, FLT_MAX, ImVec2(0, 80));
            ImGui::PlotHistogram("AverageScore", m_AverageScoreArray.data(), static_cast<int>(m_AverageScoreArray.size()), 0, NULL, FLT_MAX, FLT_MAX, ImVec2(0, 80));

            ImGui::Text("BestGenome:");
            ImGui::Indent();
            ImGui::Text("Neurons: %d", m_BestGenome.GetNeuronsCount());
            ImGui::Text("Links: %d", m_BestGenome.GetLinksCount());
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

            BrainFramework::LayeredNeuralNetwork neuralNetwork;
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

            std::unique_ptr<BrainFramework::LayeredNeuralNetwork> basicNeuralNetwork = std::make_unique<BrainFramework::LayeredNeuralNetwork>();
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
