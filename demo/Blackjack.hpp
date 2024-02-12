#pragma once

#include "../src/BrainFramework.hpp"

class Blackjack : public BrainFramework::Simulation
{
public:
    static constexpr int k_Inputs = 10;
    static constexpr int k_Outputs = 1;

    Blackjack() = default;
    Blackjack(const Blackjack&) = delete;
    Blackjack& operator=(const Blackjack&) = delete;

    void DisplayImGui() override
    {
        ImGui::Text("Hand: %d", m_Hand);
        ImGui::Text("Cards: %d", m_Cards);
        ImGui::Unindent();

        ImGui::Spacing();

        ImGui::Text("Inputs:");
        ImGui::Indent();
        for (auto input : GetInputs())
        {
            ImGui::Text("%f", input);
        }
        ImGui::Unindent();

        ImGui::Spacing();

        ImGui::Text("Outputs:");
        ImGui::Indent();
        for (auto output : GetOutputs())
        {
            ImGui::Text("%f", output);
        }
    }

    bool Initialize(BrainFramework::NeuralNetwork& neuralNetwork) override
    {
        m_Inputs.resize(k_Inputs);
        for (int i = 0; i < k_Inputs; ++i)
        {
            m_Inputs[i] = 0.0f;
        }
        m_Outputs.resize(k_Outputs);
        m_Outputs[0] = 0.0f;

        int c1 = PickCard();
        int c2 = PickCard();
        m_Inputs[0] = static_cast<float>(c1);
        m_Inputs[1] = static_cast<float>(c2);

        m_Hand = c1 + c2;
        m_Cards = 2;

        return Simulation::Initialize(neuralNetwork);
    }

    Result Step(BrainFramework::NeuralNetwork& neuralNetwork) override
    {
        if (m_Hand > 21)
        {
            AddReward(-10.0f);
            return MarkResult(Result::Finished);
        }

        const bool result = neuralNetwork.Evaluate(m_Inputs, m_Outputs);
        if (!result)
            return MarkResult(Result::Failed);

        if (m_Outputs[0] > 0.0f)
        {
            int card = PickCard();
            m_Inputs[m_Cards] = static_cast<float>(card);
            m_Hand += card;
            m_Cards++;

            if (m_Cards >= k_Inputs)
            {
                AddReward(1.0f);
                return MarkResult(Result::Finished);
            }

            if (m_Hand > 21)
            {
                AddReward(-10.0f);
                return MarkResult(Result::Finished);
            }
            else if (m_Hand == 21)
            {
                AddReward(10.0f);
                return MarkResult(Result::Finished);
            }
            else
            {
                AddReward(1.0f);
            }
        }
        else
        {
            AddReward(3.0f);
            return MarkResult(Result::Finished);
        }

        return MarkResult(Result::Ongoing);
    }

    int PickCard()
    {
        return rand() % 11 + 1;
    }

    int GetInputsCount() const override { return k_Inputs; }
    int GetOutputsCount() const override { return k_Outputs; }

    const std::vector<float>& GetInputs() const { return m_Inputs; }
    const std::vector<float>& GetOutputs() const { return m_Outputs; }

    const char* GetName() const override { return "Blackjack"; }

private:
    std::vector<float> m_Inputs;
    std::vector<float> m_Outputs;
    int m_Hand{ 0 };
    int m_Cards{ 0 };
};