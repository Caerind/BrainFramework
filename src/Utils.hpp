#pragma once

#include <cmath>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <functional>
#include <algorithm>
#include <queue>
#include <random>

namespace BrainFramework
{

static std::random_device rd;
static std::mt19937 gen(rd());

inline float Sigmoid(float x)
{
    return 2.0f / (1.0f + std::exp(-4.9f * x)) - 1.0f;
}

template <typename T>
inline int RandomIndex(const T& container)
{
    static std::random_device rd;
    static std::mt19937 gen(rd());

    if (container.empty()) 
    {
        return -1;
    }

    std::uniform_int_distribution<int> distribution(0, static_cast<int>(container.size() - 1));
    return distribution(gen);
}

inline float RandomFloat(float min = 0.0f, float max = 1.0f)
{
    std::uniform_real_distribution<float> distribution(min, max);
    return distribution(gen);
}

} // namespace BrainFramework
