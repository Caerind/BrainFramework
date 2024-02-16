#pragma once

#include "NeuralNetwork.hpp"
#include "AgentInterface.hpp"

namespace BrainFramework
{

class ISimulation
{
public:
    ISimulation() = default;
    ISimulation(const ISimulation&) = delete;
    ISimulation& operator=(const ISimulation&) = delete;

    virtual const char* GetName() const = 0;

    virtual AgentInterface* CreateRLAgent(BrainFramework::NeuralNetwork& neuralNetwork) = 0;
    virtual void RemoveAgent(AgentInterface* agent) = 0;
    virtual int GetRLInputsCount() const = 0;
    virtual int GetRLOutputsCount() const = 0;
};

template <typename BaseAgentType>
class Simulation : public ISimulation
{
public:
    Simulation() = default;
    Simulation(const Simulation&) = delete;
    Simulation& operator=(const Simulation&) = delete;

    AgentInterface* CreateRLAgent(BrainFramework::NeuralNetwork& neuralNetwork) override
    {
        return nullptr;
    }

    void RemoveAgent(AgentInterface* agent) override
    {
        m_Agents.erase(
            std::remove_if(m_Agents.begin(), m_Agents.end(),
                [agent](const std::unique_ptr<BaseAgentType*>& ptr)
                {
                    return *ptr == agent;
                }
            ),
            m_Agents.end()
        );
    }

    int GetRLInputsCount() const override { return 0; }
    int GetRLOutputsCount() const override { return 0; }

protected:
    std::vector<std::unique_ptr<BaseAgentType>> m_Agents;
};

} // namespace BrainFramework
