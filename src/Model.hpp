#pragma once

#include "Simulation.hpp"
#include "NeuralNetwork.hpp"

namespace BrainFramework
{

class Model
{
public:
    virtual const char* GetName() const = 0;

    virtual void DisplayImGui() {};

    virtual bool StartTraining(const Simulation& simulation) { m_IsTraining = true; return true; }
    virtual bool Train(Simulation& simulation) = 0;
    virtual bool StopTraining(const Simulation& simulation) { m_IsTraining = false; return true; }
    bool IsTraining() const { return m_IsTraining; }

    virtual bool MakeBestNeuralNetwork(NeuralNetwork& neuralNetwork) = 0;

private:
    bool m_IsTraining{ false };
};

} // namespace BrainFramework
