#include "Application.hpp"

#include <ctime>
#include <memory>

#include "../src/BrainFramework.hpp"
#include "NEAT.hpp"
#include "MoreOrLess.hpp"

int main()
{
    std::srand(std::time(nullptr));

    std::unique_ptr<Model> modelPtr = nullptr;
    std::unique_ptr<Simulation> simulationPtr = nullptr;

    Application app;
    app.Run([&]()
    {  
        if (ImGui::Begin("BrainFramework"))
        {
            ImGui::Text("Model: ");
            ImGui::SameLine();
            if (modelPtr == nullptr)
            {
                if (ImGui::Button("NEATModel"))
                {
                    modelPtr = std::make_unique<NEATModel>();
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
                if (simulationPtr->GetResult() == Simulation::Result::None)
                {

                }
            }


            if (Model* model = modelPtr.get())
            {
                if (Simulation* simulation = simulationPtr.get())
                {
                    if (ImGui::Button("Step"))
                    {
                        simulation->Step(*model);
                    }

                    if (MoreOrLess* moreOrLess = dynamic_cast<MoreOrLess*>(simulation))
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
                    for (auto input : simulation->GetInputs())
                    {
                        ImGui::Text("%f", input);
                    }
                    ImGui::Unindent();

                    ImGui::Spacing();

                    ImGui::Text("Outputs:");
                    ImGui::Indent();
                    for (auto output : simulation->GetOutputs())
                    {
                        ImGui::Text("%f", output);
                    }
                    ImGui::Unindent();
                }
                else
                {
                    if (ImGui::Button("Init"))
                    {
                        simulationPtr = std::make_unique<MoreOrLess>();
                        simulationPtr->Initialize(*model);
                    }
                }
            }
        }
        ImGui::End();
    });

    return 0;
}