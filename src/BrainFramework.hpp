#pragma once

#include "Utils.hpp"
#include "NeuralNetwork.hpp"
#include "Simulation.hpp"

class Model
{
public:
    virtual const char* GetName() const = 0;

    virtual void Train(Simulation& simulation) = 0;
    virtual bool MakeNeuralNetwork(NeuralNetwork& neuralNetwork) = 0;
};