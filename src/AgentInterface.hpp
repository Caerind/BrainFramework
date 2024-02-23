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
    virtual Result Step(bool allowLog = false) = 0;

    void AddLog(const std::string& log) { m_Logs.push_back(log); }

    float GetScore() const { return m_Score; }
    Result GetResult() const { return m_Result; }
    const std::vector<std::string>& GetLogs() const { return m_Logs; }

protected:
    void AddReward(float reward) { m_Score += reward; }
    Result MarkResult(Result result) { m_Result = result; return result; }

    std::vector<std::string> m_Logs;
    Result m_Result{ Result::None };
    float m_Score{ 0.0f };
};

} // namespace BrainFramework