#include "Application.hpp"

#include <ctime>
#include <memory>
#include <iostream>

#include "../src/BrainFramework.hpp"
#include "NEAT.hpp"
#include "NEET.hpp"
#include "MoreOrLess.hpp"

int main()
{
    std::srand(static_cast<unsigned int>(std::time(nullptr)));

    std::unique_ptr<BrainFramework::Model> modelPtr = nullptr;
    std::unique_ptr<BrainFramework::Simulation> simulationPtr = nullptr;
    BrainFramework::NeuralNetwork neuralNetwork;

    int trainingSteps = 1000;
    bool isTraining = false;
    bool isPlaying = false;

    Application app;
    app.Run([&]()
    {  
        if (ImGui::Begin("BrainFramework"))
        {
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
            }
            else
            {
                ImGui::Text("%s", modelPtr->GetName());
            }

            ImGui::Text("Simulation: ");
            ImGui::SameLine();
            if (simulationPtr == nullptr)
            {
                if (ImGui::Button("MoreOrLess"))
                {
                    simulationPtr = std::make_unique<MoreOrLess>();
                }
            }
            else
            {
                ImGui::Text("%s", simulationPtr->GetName());
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
                            modelPtr->StartTraining(*simulationPtr);
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
                            modelPtr->StopTraining(*simulationPtr);
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

                            modelPtr->MakeBestNeuralNetwork(neuralNetwork);
                            simulationPtr->Initialize(neuralNetwork);
                        }
                    }
                    else
                    {
                        if (ImGui::Button("Stop playing"))
                        {
                            isPlaying = false;
                        }

                        if (simulationPtr->GetResult() == BrainFramework::Simulation::Result::Ongoing || simulationPtr->GetResult() == BrainFramework::Simulation::Result::Initialized)
                        {
                            if (ImGui::Button("Step"))
                            {
                                simulationPtr->Step(neuralNetwork);
                            }
                        }

                        if (simulationPtr->GetResult() == BrainFramework::Simulation::Result::Finished || simulationPtr->GetResult() == BrainFramework::Simulation::Result::Failed)
                        {
                            if (ImGui::Button("Stop"))
                            {
                                isPlaying = false;
                            }
                        }

                        ImGui::Text("%s:", simulationPtr->GetName());
                        ImGui::Indent();
                        simulationPtr->DisplayImGui();
                        ImGui::Unindent();
                    }
                }
            }
        }
        ImGui::End();
    });

    return 0;
}
