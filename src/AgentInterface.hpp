#pragma once

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

    float GetScore() const { return m_Score; }
    Result GetResult() const { return m_Result; }

protected:
    void AddReward(float reward) { m_Score += reward; }
    Result MarkResult(Result result) { m_Result = result; return result; }

    Result m_Result{ Result::None };
    float m_Score{ 0.0f };
};

} // namespace BrainFramework