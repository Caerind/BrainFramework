#pragma once

#include <cmath>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <functional>
#include <algorithm>
#include <queue>

namespace BrainFramework
{

inline float Sigmoid(float x)
{
    return 2.0f / (1.0f + std::exp(-4.9f * x)) - 1.0f;
}

} // namespace BrainFramework
