#pragma once

#include "Simulation.hpp"
#include "NeuralNetwork.hpp"

namespace BrainFramework
{

class Model
{
public:
    virtual const char* GetName() const = 0;

    virtual bool StartTraining() { m_IsTraining = true; return true; }
    virtual void Train(Simulation& simulation) = 0;
    virtual bool StopTraining() { m_IsTraining = false; return true; }
    bool IsTraining() const { return m_IsTraining; }

    virtual bool MakeBestNeuralNetwork(NeuralNetwork& neuralNetwork) = 0;

private:
    bool m_IsTraining{ false };
};

} // namespace BrainFramework
