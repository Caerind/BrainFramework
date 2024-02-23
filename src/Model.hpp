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

    virtual bool PrepareTraining(const ISimulation& simulation) { return true; }
    virtual bool StartEvaluation(std::unique_ptr<NeuralNetwork>& neuralNetwork) = 0;
    virtual bool EndEvalutation(float result) = 0;

    virtual bool MakeBestNeuralNetwork(std::unique_ptr<NeuralNetwork>& neuralNetwork, int index = 0) = 0;
};

} // namespace BrainFramework