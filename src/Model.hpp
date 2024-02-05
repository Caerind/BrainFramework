#pragma once

#include "Simulation.hpp"
#include "NeuralNetwork.hpp"

namespace BrainFramework
{

class Model
{
public:
    virtual const char* GetName() const = 0;

    virtual void Train(Simulation& simulation) = 0;
    virtual bool MakeBestNeuralNetwork(NeuralNetwork& neuralNetwork) = 0;
};

} // namespace BrainFramework
