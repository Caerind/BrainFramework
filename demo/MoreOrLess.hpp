#pragma once

#include "../src/BrainFramework.hpp"

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

    int GetGuessedNumber() const override { return static_cast<int>(std::round(m_Outputs[0] * 100.0f)); }

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
    using super = BrainFramework::Simulation<MoreOrLessBaseAgent>;

    MoreOrLess()
        : super(BrainFramework::AgentCountSettings(BrainFramework::AgentCountSettings::Max, 4))
    {
    }
    MoreOrLess(const MoreOrLess&) = delete;
    MoreOrLess& operator=(const MoreOrLess&) = delete;

    void Initialize() override
    {
        m_NumberToGuess = std::rand() % (100 + 1); // [0,100]
    }

    bool IsFinished() const override { return false; }

    int GetNumberToGuess() const
    {
        return m_NumberToGuess;  
    }

    const char* GetName() const override { return "MoreOrLess"; }

    bool CanTrainRL() const { return true; }
    int GetRLInputsCount() const override { return MoreOrLessRLAgent::k_Inputs; }
    int GetRLOutputsCount() const override { return MoreOrLessRLAgent::k_Outputs; }
    BrainFramework::AgentInterface* CreateRLAgent(BrainFramework::NeuralNetwork& neuralNetwork) override
    {
        return CreateAgent<MoreOrLessRLAgent>(*this, neuralNetwork);
    }

private:
    int m_NumberToGuess {-1};
};

void MoreOrLessBaseAgent::Initialize()
{
    m_Guess = 0;
    m_Result = BrainFramework::AgentInterface::Result::Initialized;
}

MoreOrLessBaseAgent::Result MoreOrLessBaseAgent::Step()
{
    if (m_Guess == 0 && HasLogger())
        AddLog("NumberToGuess: " + std::to_string(m_MoreOrLess.GetNumberToGuess()));

    if (m_Guess < 10)
    {
        if (!Evaluate())
            return MarkResult(Result::Failed);

        int numberGuessed = GetGuessedNumber();

        static std::string log;
        if (HasLogger())
        {
            log.clear();
            log.reserve(16);
            log += std::to_string(numberGuessed);
        }

        if (numberGuessed < 0 || numberGuessed > 100)
        {
            AddReward(-100.0f);
        }
        else if (numberGuessed > 0)
        {
            AddReward(0.1f);
        }

        if (numberGuessed == m_MoreOrLess.GetNumberToGuess())
        {
            AddReward(100.0f / (m_Guess + 1));
            AddLog(log);
            return MarkResult(Result::Finished);
        }
        else
        {
            const float hint = numberGuessed > m_MoreOrLess.GetNumberToGuess() ? -1.0f : 1.0f;

            AddFeedback(m_Guess, numberGuessed, hint);

            if (HasLogger() && hint > 0.0f)
                log += " +";
            if (HasLogger() && hint < 0.0f)
                log += " -";

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

        AddLog(log);
    }

    if (m_Guess >= 10)
    {
        AddReward(-10.0f);
        return MarkResult(Result::Finished);
    }

    return MarkResult(Result::Ongoing);
}