#include "Application.hpp"

#include <ctime>
#include <memory>

#include "../src/BrainFramework.hpp"
#include "NEAT.hpp"
#include "MoreOrLess.hpp"

int main()
{
    std::srand(std::time(nullptr));

    std::unique_ptr<BrainFramework::Model> modelPtr = nullptr;
    std::unique_ptr<BrainFramework::Simulation> simulationPtr = nullptr;
    BrainFramework::NeuralNetwork neuralNetwork;

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
                    modelPtr = std::make_unique<NEAT>();
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
                        if (ImGui::Button("Train"))
                        {
                            modelPtr->StartTraining();
                            isTraining = true;
                        }
                    }
                    else
                    {
                        modelPtr->Train(*simulationPtr);

                        if (ImGui::Button("Stop training"))
                        {
                            modelPtr->StopTraining();
                            isTraining = false;
                        }
                    }
                }

                if (!isTraining)
                {
                    if (!isPlaying)
                    {
                        if (ImGui::Button("Play"))
                        {
                            isPlaying = true;
                        }
                    }
                    else
                    {
                        if (ImGui::Button("Stop playing"))
                        {
                            isPlaying = false;
                        }

                        if (ImGui::Button("Step"))
                        {
                            if (simulationPtr->GetResult() == BrainFramework::Simulation::Result::None)
                            {
                                modelPtr->MakeBestNeuralNetwork(neuralNetwork);
                                simulationPtr->Initialize(neuralNetwork);
                            }

                            simulationPtr->Step(neuralNetwork);
                        }

                        if (MoreOrLess* moreOrLess = dynamic_cast<MoreOrLess*>(simulationPtr.get()))
                        {
                            ImGui::Text("MoreOrLess:");
                            ImGui::Indent();
                            ImGui::Text("NumberToGuess: %d", moreOrLess->GetNumberToGuess());
                            ImGui::Text("GuessCount: %d", moreOrLess->GetGuess());
                            ImGui::Unindent();

                            ImGui::Spacing();
                        }

                        ImGui::Text("Inputs:");
                        ImGui::Indent();
                        for (auto input : simulationPtr->GetInputs())
                        {
                            ImGui::Text("%f", input);
                        }
                        ImGui::Unindent();

                        ImGui::Spacing();

                        ImGui::Text("Outputs:");
                        ImGui::Indent();
                        for (auto output : simulationPtr->GetOutputs())
                        {
                            ImGui::Text("%f", output);
                        }
                        ImGui::Unindent();
                    }
                }
            }
        }
        ImGui::End();
    });

    return 0;
}
