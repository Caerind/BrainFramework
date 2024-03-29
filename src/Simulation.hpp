#pragma once

#include "NeuralNetwork.hpp"
#include "AgentInterface.hpp"

namespace BrainFramework
{

class AgentCountSettings
{
public:
    enum AgentCountType
    {
        Fixed,
        Max,
        Unlimited
    };

    AgentCountSettings(AgentCountType type, int value = -1) : m_Type(type), m_Value(value) {}

    AgentCountType GetAgentCountType() const { return m_Type; }
    int GetValue() const { return m_Type != AgentCountType::Unlimited ? m_Value : -1; }

private:
    AgentCountType m_Type;
    int m_Value;
};

class ISimulation
{
public:
    ISimulation(const AgentCountSettings& agentCountSettings) : m_AgentCountSettings(agentCountSettings) {}
    ISimulation(const ISimulation&) = delete;
    ISimulation& operator=(const ISimulation&) = delete;

    virtual const char* GetName() const = 0;

    virtual void Initialize() {}
    virtual bool IsFinished() const = 0;

    virtual bool CanTrainRL() const = 0;
    virtual int GetRLInputsCount() const = 0;
    virtual int GetRLOutputsCount() const = 0;
    virtual AgentInterface* CreateRLAgent(BrainFramework::NeuralNetwork& neuralNetwork) = 0;

    virtual void RemoveAgent(AgentInterface* agent) = 0;

    const AgentCountSettings& GetAgentCountSettings() const { return m_AgentCountSettings; }

private:
    AgentCountSettings m_AgentCountSettings;
};

template <typename BaseAgentType>
class Simulation : public ISimulation
{
public:
    Simulation(const AgentCountSettings& agentCountSettings) : ISimulation(agentCountSettings) {}
    Simulation(const Simulation&) = delete;
    Simulation& operator=(const Simulation&) = delete;

    bool CanTrainRL() const override { return false; }
    int GetRLInputsCount() const override { return 0; }
    int GetRLOutputsCount() const override { return 0; }
    AgentInterface* CreateRLAgent(BrainFramework::NeuralNetwork& neuralNetwork) override
    {
        return nullptr;
    }

    template <typename AgentType, typename ... AgentArgs>
    BaseAgentType* CreateAgent(AgentArgs&& ... args)
    {
        m_Agents.emplace_back(std::make_unique<AgentType>(std::forward<AgentArgs>(args)...));
        return m_Agents.back().get();
    }

    void RemoveAgent(AgentInterface* agent) override
    {
        m_Agents.erase(
            std::remove_if(m_Agents.begin(), m_Agents.end(),
                [agent](const std::unique_ptr<BaseAgentType>& ptr)
                {
                    return static_cast<AgentInterface*>(ptr.get()) == agent;
                }
            ),
            m_Agents.end()
        );
    }

private:
    std::vector<std::unique_ptr<BaseAgentType>> m_Agents;
};

} // namespace BrainFramework
