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

struct Player
{
    Player() = default;

    std::unique_ptr<BrainFramework::Model> model;
    std::unique_ptr<BrainFramework::NeuralNetwork> neuralNetwork;
    BrainFramework::AgentInterface* agent;
};

enum class State
{
    Config,
    Menu,
    Train,
    Play
};

int main()
{
    std::srand(static_cast<unsigned int>(std::time(nullptr)));

    std::unique_ptr<BrainFramework::ISimulation> simulationPtr = nullptr;
    std::vector<Player> players;

    State state = State::Config;

    int trainingSteps = 10;
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

            if (simulationPtr != nullptr)
            {
                int currentPlayerCount = static_cast<int>(players.size());

                int minPlayerCount = 1;
                if (simulationPtr->GetAgentCountSettings().GetAgentCountType() == BrainFramework::AgentCountSettings::Fixed)
                {
                    minPlayerCount = simulationPtr->GetAgentCountSettings().GetValue();
                }

                int maxPlayerCount = simulationPtr->GetAgentCountSettings().GetValue();
                if (simulationPtr->GetAgentCountSettings().GetAgentCountType() == BrainFramework::AgentCountSettings::Unlimited)
                {
                    maxPlayerCount = 10;
                }

                for (Player& player : players)
                {
                    switch (state)
                    {
                        case State::Config:
                        {
                            // TODO : Model config
                            ImGui::Text("%s", player.model->GetName());
                        } break;
                        case State::Menu: break;
                        case State::Train:
                        {
                            if (ImGui::CollapsingHeader(player.model->GetName()))
                            {
                                player.model->DisplayImGui();
                            }
                        } break;
                        case State::Play:
                        {
                            // Nothing ?
                        } break;
                    }
                }

                switch (state)
                {
                case State::Config:
                {
                    if (currentPlayerCount < maxPlayerCount)
                    {
                        if (ImGui::Button("NEAT"))
                        {
                            Player& player = players.emplace_back();
                            player.model = std::make_unique<NEAT::NEATModel>();
                        }
                        ImGui::SameLine();
                        if (ImGui::Button("NEET"))
                        {
                            Player& player = players.emplace_back();
                            player.model = std::make_unique<NEET::NEETModel>();
                        }
                        ImGui::SameLine();
                        if (ImGui::Button("NEETL"))
                        {
                            Player& player = players.emplace_back();
                            player.model = std::make_unique<NEETL::NEETLModel>();
                        }
                    }

                    if (currentPlayerCount >= minPlayerCount)
                    {
                        if (ImGui::Button("Ready"))
                        {
                            state = State::Menu;
                        }
                    }
                } break;

                case State::Menu:
                {
                    if (ImGui::Button("Train"))
                    {
                        state = State::Train;
                        for (Player& player : players)
                            player.model->PrepareTraining(*simulationPtr);
                    }
                    ImGui::SameLine();
                    if (ImGui::Button("Play"))
                    {
                        state = State::Play;

                        simulationPtr->Initialize();

                        for (Player& player : players)
                        {
                            player.model->MakeBestNeuralNetwork(player.neuralNetwork);
                            player.agent = simulationPtr->CreateRLAgent(*player.neuralNetwork);
                            player.agent->Initialize();
                        }
                    }
                } break;

                case State::Train:
                {
                    ImGui::InputInt("TrainingSteps", &trainingSteps);
                    if (trainingSteps < 1)
                        trainingSteps = 1;

                    if (ImGui::Button("Stop training"))
                    {
                        state = State::Menu;
                    }

                    for (int trainingStep = 0; trainingStep < trainingSteps; ++trainingStep)
                    {
                        simulationPtr->Initialize();

                        for (Player& player : players)
                        {
                            player.model->StartEvaluation(player.neuralNetwork);
                            player.agent = simulationPtr->CreateRLAgent(*player.neuralNetwork);
                            player.agent->Initialize();
                        }

                        bool simualtionShouldContinue = true;
                        bool someAgentIsStillPlaying = true;

                        do
                        {
                            simualtionShouldContinue = !simulationPtr->IsFinished();
                            someAgentIsStillPlaying = false;

                            for (Player& player : players)
                            {
                                auto result = player.agent->Step();
                                if (result == BrainFramework::AgentInterface::Result::Ongoing)
                                {
                                    someAgentIsStillPlaying = true;
                                }
                            }

                        } while (simualtionShouldContinue && someAgentIsStillPlaying);

                        for (Player& player : players)
                        {
                            player.model->EndEvalutation(player.agent->GetScore());
                            simulationPtr->RemoveAgent(player.agent); // TODO : Reset agent on Initialize ?
                            player.agent = nullptr;
                        }
                    }

                } break;

                case State::Play:
                {
                    if (ImGui::Button("Stop playing"))
                    {
                        state = State::Menu;
                    }

                    if (ImGui::Button("Step"))
                    {
                        bool simualtionShouldContinue = !simulationPtr->IsFinished();
                        bool someAgentIsStillPlaying = false;

                        for (Player& player : players)
                        {
                            if (player.agent != nullptr)
                            {
                                auto result = player.agent->Step(true);
                                if (result == BrainFramework::AgentInterface::Result::Ongoing)
                                {
                                    someAgentIsStillPlaying = true;
                                }
                            }
                        }

                        if (!(simualtionShouldContinue && someAgentIsStillPlaying))
                        {
                            for (Player& player : players)
                            {
                                player.model->EndEvalutation(player.agent->GetScore());
                                simulationPtr->RemoveAgent(player.agent); // TODO : Reset agent on Initialize ?
                                player.agent = nullptr;
                            }
                        }
                    }

                    ImGui::Separator();

                    for (Player& player : players)
                    {
                        if (player.agent != nullptr)
                        {
                            for (const std::string& log : player.agent->GetLogs())
                            {
                                ImGui::Text("%s", log.c_str());
                            }
                        }

                        ImGui::Separator();
                    }
                    
                } break;
                }
            }
        }
        ImGui::End();
    });

    return 0;
}
