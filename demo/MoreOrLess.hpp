#pragma once

#include "../src/BrainFramework.hpp"
#include <string>

class MoreOrLess;

class MoreOrLessBaseAgent : public BrainFramework::AgentInterface
{
public:
    MoreOrLessBaseAgent(MoreOrLess& moreOrLess)
        : m_MoreOrLess(moreOrLess)
    {
    }

    void Initialize() override;
    Result Step() override;

    virtual bool Evaluate() = 0;
    virtual int GetGuessedNumber() const = 0;
    virtual void AddFeedback(int guessCount, int numberGuessed, float hint) = 0;

private:
    MoreOrLess& m_MoreOrLess;
protected:
    int m_NumberToGuess{ 0 };
    int m_Guess{ 0 };
    int m_PreviousGuessed{ -1 };
    float m_PreviousHint{ 0.0f };
};

class MoreOrLessRLAgent : public MoreOrLessBaseAgent
{
public:
    static constexpr int k_Inputs = 20;
    static constexpr int k_Outputs = 1;

    MoreOrLessRLAgent(MoreOrLess& moreOrLess, BrainFramework::NeuralNetwork& neuralNetwork)
        : MoreOrLessBaseAgent(moreOrLess)
        , m_NeuralNetwork(neuralNetwork)
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
    }

    MoreOrLessRLAgent(const MoreOrLessRLAgent&) = delete;
    MoreOrLessRLAgent& operator=(const MoreOrLessRLAgent&) = delete;

    bool Evaluate() override
    {
        return m_NeuralNetwork.Evaluate(m_Inputs, m_Outputs);
    }

    int GetGuessedNumber() const override { static_cast<int>(std::round(m_Outputs[0] * 100.0f)); }

    void AddFeedback(int guessCount, int numberGuessed, float hint) override
    {
        const int index = guessCount * 2;
        m_Inputs[index] = static_cast<float>(numberGuessed);
        m_Inputs[index + 1] = hint;
    }

private:
    BrainFramework::NeuralNetwork& m_NeuralNetwork;
    std::vector<float> m_Inputs;
    std::vector<float> m_Outputs;
};

class MoreOrLess : public BrainFramework::Simulation<MoreOrLessBaseAgent>
{
public:
    MoreOrLess() = default;
    MoreOrLess(const MoreOrLess&) = delete;
    MoreOrLess& operator=(const MoreOrLess&) = delete;

    int GetNumberToGuess() const
    {
        return std::rand() % (100 + 1); // [0,100]
    }

    const char* GetName() const override { return "MoreOrLess"; }

    BrainFramework::AgentInterface* CreateRLAgent(BrainFramework::NeuralNetwork& neuralNetwork) override
    {
        m_Agents.emplace_back(std::make_unique<MoreOrLessRLAgent>(*this, neuralNetwork));
        return m_Agents.back().get();
    }

    int GetRLInputsCount() const override { return MoreOrLessRLAgent::k_Inputs; }
    int GetRLOutputsCount() const override { return MoreOrLessRLAgent::k_Outputs; }
};



void MoreOrLessBaseAgent::Initialize()
{
    m_NumberToGuess = m_MoreOrLess.GetNumberToGuess();
    m_Guess = 0;
}

MoreOrLessBaseAgent::Result MoreOrLessBaseAgent::Step()
{
    if (m_Guess < 10)
    {
        if (!Evaluate())
            return MarkResult(Result::Failed);

        int numberGuessed = GetGuessedNumber();

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
            const float hint = numberGuessed > m_NumberToGuess ? -1.0f : 1.0f;

            AddFeedback(m_Guess, numberGuessed, hint);

            if (m_Guess > 0)
            {
                if ((m_PreviousHint > 0.0f && numberGuessed > m_PreviousGuessed) || (m_PreviousHint < 0.0f && numberGuessed < m_PreviousGuessed))
                {
                    AddReward(1.0f);
                }
            }

            m_Guess++;
            m_PreviousGuessed = numberGuessed;
            m_PreviousHint = hint;
        }
    }

    if (m_Guess >= 10)
    {
        AddReward(-10.0f);
        return MarkResult(Result::Finished);
    }

    return MarkResult(Result::Ongoing);
}