#pragma once

#include "Utils.hpp"

namespace BrainFramework
{

// TODO : Split in different to allow Agent which are not RL agents
class AgentInterface
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

    virtual void Initialize() = 0;
    virtual Result Step() = 0;

    void SetLogger(Logger* logger) { m_Logger = logger; }
    bool HasLogger() const { return m_Logger != nullptr; }
    void AddLog(const std::string& line) 
    { 
        if (m_Logger != nullptr)
        {
            m_Logger->Log(line);
        }
    }

    float GetGameScore() const { return m_GameScore; }
    float GetReward() const { return m_Reward; }
    Result GetResult() const { return m_Result; }

protected:
    void AddReward(float reward) { m_Reward += reward; }
    void SetGameScore(float gameScore) { m_GameScore = gameScore; }
    Result MarkResult(Result result) { m_Result = result; return result; }

    Logger* m_Logger{ nullptr };
    Result m_Result{ Result::None };
    float m_GameScore{ 0.0f };
    float m_Reward{ 0.0f };
};

} // namespace BrainFramework