#pragma once

#include <cmath>
#include <vector>
#include <unordered_map>
#include <functional>
#include <algorithm>
#include <queue>

constexpr int k_Population = 300;
constexpr float k_DeltaDisjoint = 2.0f;
constexpr float k_DeltaWeights = 0.4f;
constexpr float k_DeltaThreshold = 1.0f;
constexpr int k_StaleSpecies = 15;
constexpr float k_MutateConnectionsChance = 0.25f;
constexpr float k_PerturbChance = 0.90f;
constexpr float k_CrossoverChance = 0.75f;
constexpr float k_LinkMutationChance = 2.0f;
constexpr float k_NodeMutationChance = 0.50f;
constexpr float k_BiasMutationChance = 0.40f;
constexpr float k_StepSize = 0.1f;
constexpr float k_DisableMutationChance = 0.4f;
constexpr float k_EnableMutationChance = 0.2f;
constexpr float k_TimeoutConstant = 20.0f;
constexpr int k_MaxNodes = 1000000;

constexpr int k_Inputs = 10;
constexpr int k_Outputs = 10;

inline float Sigmoid(float x)
{
    return 2.0f / (1.0f + std::exp(-4.9f * x)) - 1.0f;
}