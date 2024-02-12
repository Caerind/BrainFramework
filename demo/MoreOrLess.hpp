#pragma once

#include "../src/BrainFramework.hpp"

#include <iostream>

class MoreOrLess : public BrainFramework::Simulation
{
public:
    static constexpr int k_Inputs = 20;
    static constexpr int k_Outputs = 1;

    MoreOrLess() = default;
    MoreOrLess(const MoreOrLess&) = delete;
    MoreOrLess& operator=(const MoreOrLess&) = delete;

    void DisplayImGui() override
    {
        ImGui::Text("NumberToGuess: %d", GetNumberToGuess());
        ImGui::Text("GuessCount: %d", GetGuess());
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
            if (i % 2 == 0)
            {
                m_Inputs[i] = -1.0f;
            }
            else
            {
                m_Inputs[i] = 0.0f;
            }
        }
        m_Outputs.resize(k_Outputs);
        m_Outputs[0] = 0.0f;

        m_NumberToGuess = std::rand() % (100 + 1); // [0,100]
        m_Guess = 0;

        return Simulation::Initialize(neuralNetwork);
    }

    Result Step(BrainFramework::NeuralNetwork& neuralNetwork) override
    {
        if (m_Guess < 10)
        {
            const bool result = neuralNetwork.Evaluate(m_Inputs, m_Outputs);
            if (!result)
                return MarkResult(Result::Failed);

            int numberGuessed = static_cast<int>(std::round(m_Outputs[0] * 100.0f));

            if (numberGuessed < 0 || numberGuessed > 100)
            {
                AddReward(-100.0f);
            }
            else if (numberGuessed > 0)
            {
                AddReward(0.1f);
            }

            if (numberGuessed == m_NumberToGuess)
            {
                AddReward(100.0f / (m_Guess + 1));
                return MarkResult(Result::Finished);
            }
            else
            {
                int index = m_Guess * 2;
                m_Inputs[index] = static_cast<float>(numberGuessed);
                m_Inputs[index + 1] = numberGuessed > m_NumberToGuess ? -1.0f : 1.0f;

                if (index > 0)
                {
                    int previousNumber = static_cast<int>(m_Inputs[index - 2]);
                    float previousHint = m_Inputs[index - 1];
                    if ((previousHint > 0.0f && numberGuessed > previousNumber) || (previousHint < 0.0f && numberGuessed < previousNumber))
                    {
                        AddReward(1.0f);
                    }
                }

                m_Guess++;
            }
        }

        if (m_Guess >= 10)
        {
            AddReward(-10.0f);
            return MarkResult(Result::Finished);
        }

        return MarkResult(Result::Ongoing);
    }

    int GetInputsCount() const override { return k_Inputs; }
    int GetOutputsCount() const override { return k_Outputs; }

    const std::vector<float>& GetInputs() const { return m_Inputs; }
    const std::vector<float>& GetOutputs() const { return m_Outputs; }

    const char* GetName() const override { return "MoreOrLess"; }

    int GetNumberToGuess() const { return m_NumberToGuess; }
    int GetGuess() const { return m_Guess; }

private:
    std::vector<float> m_Inputs;
    std::vector<float> m_Outputs;

    int m_NumberToGuess{ 0 };
    int m_Guess{ 0 };
};