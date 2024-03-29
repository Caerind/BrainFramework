#pragma once

#include "../src/BrainFramework.hpp"

class Blackjack;

class BlackjackBaseAgent : public BrainFramework::AgentInterface
{
public:
    BlackjackBaseAgent(Blackjack& blackjack)
        : m_Blackjack(blackjack)
    {
    }

    void Initialize() override;
    Result Step() override;

    virtual bool Evaluate() = 0;
    virtual void AddCard(int card) { m_Hand += card; m_Cards++; }
    virtual bool TakeCard() = 0;

    int GetHand() const { return m_Hand; }
    int GetCardsCount() const { return m_Cards; }

private:
    Blackjack& m_Blackjack;
protected:
    int m_Hand{ 0 };
    int m_Cards{ 0 };
};

class BlackjackRLAgent : public BlackjackBaseAgent
{
public:
    static constexpr int k_Inputs = 20;
    static constexpr int k_Outputs = 1;

    BlackjackRLAgent(Blackjack& blackjack, BrainFramework::NeuralNetwork& neuralNetwork)
        : BlackjackBaseAgent(blackjack)
        , m_NeuralNetwork(neuralNetwork)
    {
        m_Inputs.resize(k_Inputs);
        for (int i = 0; i < k_Inputs; ++i)
        {
            m_Inputs[i] = 0.0f;
        }
        m_Outputs.resize(k_Outputs);
        m_Outputs[0] = 0.0f;
        m_Cards = 0;
    }

    bool Evaluate() override
    {
        return m_NeuralNetwork.Evaluate(m_Inputs, m_Outputs);
    }

    void AddCard(int card) override
    {
        m_Inputs[m_Cards] = static_cast<float>(card);
        m_Cards++;
    }

    bool TakeCard() override { return m_Outputs[0] >= 0.0f; }

private:
    BrainFramework::NeuralNetwork& m_NeuralNetwork;
    std::vector<float> m_Inputs;
    std::vector<float> m_Outputs;
    int m_Cards{ 0 };
};

class Blackjack : public BrainFramework::Simulation<BlackjackBaseAgent>
{
public:
    using super = BrainFramework::Simulation<BlackjackBaseAgent>;

    Blackjack()
        : super(BrainFramework::AgentCountSettings(BrainFramework::AgentCountSettings::Max, 4))
    {
    }
    Blackjack(const Blackjack&) = delete;
    Blackjack& operator=(const Blackjack&) = delete;

    void Initialize() override
    {
        m_AllCards.clear();
        for (int color = 0; color < 4; ++color)
        {
            for (int i = 2; i < 10; ++i)
            {
                m_AllCards.push_back(i); // 2 to 9
            }

            m_AllCards.push_back(10);
            m_AllCards.push_back(10);
            m_AllCards.push_back(10);

            m_AllCards.push_back(1); // As is tricky...
        }
    }

    bool IsFinished() const override { return false; }

    int PickCard()
    {
        const int rIndex = BrainFramework::RandomIndex(m_AllCards);
        const int card = m_AllCards[rIndex];
        m_AllCards.erase(m_AllCards.begin() + rIndex);
        return card;
    }

    const char* GetName() const override { return "Blackjack"; }

    bool CanTrainRL() const { return true; }
    int GetRLInputsCount() const override { return BlackjackRLAgent::k_Inputs; }
    int GetRLOutputsCount() const override { return BlackjackRLAgent::k_Outputs; }
    BrainFramework::AgentInterface* CreateRLAgent(BrainFramework::NeuralNetwork& neuralNetwork) override
    {
        return CreateAgent<BlackjackRLAgent>(*this, neuralNetwork);
    }

private: 
    std::vector<int> m_AllCards;
};


void BlackjackBaseAgent::Initialize()
{ 
    m_Hand = 0; 
    m_Cards = 0;
    m_Result = BrainFramework::AgentInterface::Result::Initialized;
}

BlackjackBaseAgent::Result BlackjackBaseAgent::Step()
{
    if (m_Hand > 21)
    {
        AddReward(-10.0f);
        return MarkResult(Result::Finished);
    }

    if (!Evaluate())
        return MarkResult(Result::Failed);

    if (TakeCard())
    {
        int card = m_Blackjack.PickCard();
        AddCard(card);
        m_Hand += card;
        m_Cards++;

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