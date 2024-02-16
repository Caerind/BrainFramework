#include "Application.hpp"

#include <ctime>
#include <memory>
#include <iostream>

#include "../src/BrainFramework.hpp"

#include "NEAT.hpp"
#include "NEET.hpp"
#include "NEETL.hpp"

#include "MoreOrLess.hpp"
#include "Blackjack.hpp"

int main()
{
    std::srand(static_cast<unsigned int>(std::time(nullptr)));

    std::unique_ptr<BrainFramework::ISimulation> simulationPtr = nullptr;
    std::unique_ptr<BrainFramework::Model> modelPtr = nullptr;
    std::unique_ptr<BrainFramework::NeuralNetwork> neuralNetworkPtr = nullptr;
    BrainFramework::AgentInterface* playingAgent = nullptr;

    int trainingSteps = 1000;
    bool isTraining = false;
    bool isPlaying = false;

    Application app;
    app.Run([&]()
    {  
        if (ImGui::Begin("BrainFramework"))
        {
            ImGui::Text("Simulation: ");
            ImGui::SameLine();
            if (simulationPtr == nullptr)
            {
                if (ImGui::Button("MoreOrLess"))
                {
                    simulationPtr = std::make_unique<MoreOrLess>();
                }
                ImGui::SameLine();
                if (ImGui::Button("Blackjack"))
                {
                    simulationPtr = std::make_unique<Blackjack>();
                }
            }
            else
            {
                ImGui::Text("%s", simulationPtr->GetName());
            }

            ImGui::Text("Model: ");
            ImGui::SameLine();
            if (modelPtr == nullptr)
            {
                if (ImGui::Button("NEAT"))
                {
                    modelPtr = std::make_unique<NEAT::NEATModel>();
                }
                ImGui::SameLine();
                if (ImGui::Button("NEET"))
                {
                    modelPtr = std::make_unique<NEET::NEETModel>();
                }
                ImGui::SameLine();
                if (ImGui::Button("NEETL"))
                {
                    modelPtr = std::make_unique<NEETL::NEETLModel>();
                }
            }
            else
            {
                ImGui::Text("%s", modelPtr->GetName());
            }

            if (modelPtr != nullptr && simulationPtr != nullptr)
            {
                if (!isPlaying)
                {
                    if (!isTraining)
                    {
                        ImGui::InputInt("TrainingSteps", &trainingSteps);
                        if (trainingSteps < 1)
                            trainingSteps = 1;

                        if (ImGui::Button("Train"))
                        {
                            modelPtr->PrepareTraining(*simulationPtr);
                            isTraining = true;
                        }
                    }
                    else
                    {
                        for (int i = 0; i < trainingSteps; ++i)
                        {
                            modelPtr->Train(*simulationPtr);
                        }

                        if (ImGui::Button("Stop training"))
                        {
                            isTraining = false;
                        }

                        ImGui::Text("%s:", modelPtr->GetName());
                        ImGui::Indent();
                        modelPtr->DisplayImGui();
                        ImGui::Unindent();
                    }
                }

                if (!isTraining)
                {
                    if (!isPlaying)
                    {
                        if (ImGui::Button("Play"))
                        {
                            isPlaying = true;
                            modelPtr->MakeBestNeuralNetwork(neuralNetworkPtr);
                            playingAgent = simulationPtr->CreateRLAgent(*neuralNetworkPtr);
                        }
                    }
                    else
                    {
                        if (ImGui::Button("Stop playing"))
                        {
                            isPlaying = false;
                        }

                        if (playingAgent->GetResult() == BrainFramework::AgentInterface::Result::Ongoing || playingAgent->GetResult() == BrainFramework::AgentInterface::Result::Initialized)
                        {
                            if (ImGui::Button("Step"))
                            {
                                playingAgent->Step();
                            }
                        }

                        if (playingAgent->GetResult() == BrainFramework::AgentInterface::Result::Finished || playingAgent->GetResult() == BrainFramework::AgentInterface::Result::Failed)
                        {
                            if (ImGui::Button("Stop"))
                            {
                                isPlaying = false;
                                simulationPtr->RemoveAgent(playingAgent);
                                playingAgent = nullptr;
                            }
                        }
                    }
                }
            }
        }
        ImGui::End();
    });

    return 0;
}
