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
    virtual bool Train(ISimulation& simulation) = 0;

    virtual bool MakeBestNeuralNetwork(std::unique_ptr<NeuralNetwork>& neuralNetwork) = 0;
};


} // namespace BrainFramework