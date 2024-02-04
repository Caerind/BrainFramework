#pragma once

#include <cmath>
#include <vector>
#include <unordered_map>
#include <functional>
#include <algorithm>
#include <queue>

inline float Sigmoid(float x)
{
    return 2.0f / (1.0f + std::exp(-4.9f * x)) - 1.0f;
}