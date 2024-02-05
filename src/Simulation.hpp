#pragma once

#include "NeuralNetwork.hpp"

namespace BrainFramework
{

class Simulation
{
public:
    enum class Result
    {
        None,
        Initialized,
        Finished,
        Failed,
        Ongoing
    };

    Simulation() = default;
    Simulation(const Simulation&) = delete;
    Simulation& operator=(const Simulation&) = delete;

    virtual bool Initialize(NeuralNetwork& neuralNetwork) { m_Score = 0.0f; m_Result = Result::Initialized; return true; };
    virtual Result Step(NeuralNetwork& neuralNetwork) = 0;

    virtual const char* GetName() const = 0;

    float GetScore() const { return m_Score; }
    Result GetResult() const { return m_Result; }

    const std::vector<float>& GetInputs() const { return m_Inputs; }
    const std::vector<float>& GetOutputs() const { return m_Outputs; }

protected:
    void AddReward(float reward) { m_Score += reward; }
    Result MarkResult(Result result) { m_Result = result; return result; }

    std::vector<float> m_Inputs;
    std::vector<float> m_Outputs;

private:
    float m_Score{ 0.0f };
    Result m_Result{ Result::None };
};

} // namespace BrainFramework
