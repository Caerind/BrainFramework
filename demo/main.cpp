#include "Application.hpp"

#include <Keys.hpp>
#include <NameGenerator.hpp>
#include <Sample.hpp>
#include <ITFile.hpp>
#include <Strategy.hpp>
#include <Generator.hpp>

#include <set>
#include <filesystem>

#include <SFML/System.hpp>
#include <SFML/Audio.hpp>

enum class Instruments
{
    Guitar,
    Bass,
    Kick,
    HHClosed,
    HHOpen,
    Snare,

    COUNT
};

struct SampleData
{
    std::unique_ptr<Sample> sample;
    sf::SoundBuffer buffer;
    sf::Sound sound;

    void DisplayImGui()
    {
        if (ImGui::Button("Play"))
            sound.play();
        ImGui::SameLine();
        if (ImGui::Button("Regen"))
        {
            Generate();
            sound.play();
        }
        ImGui::SameLine();
        if (ImGui::CollapsingHeader(sample->GetName().c_str()))
        {
            ImGui::Indent();
            bool changed = sample->DisplayImGui();
            if (changed)
            {
                Generate();
                sound.play();
            }
            ImGui::Unindent();
        }
    }

    void Generate()
    {
        sample->Generate();

        const std::vector<float>& data = sample->GetData();

        std::vector<sf::Int16> sampleValues;
        sampleValues.reserve(data.size());
        constexpr sf::Int16 min16 = -0x8000;
        constexpr sf::Int16 max16 = 0x7FFF;
        for (const auto& value : data)
        {
            sampleValues.push_back(static_cast<std::int16_t>(std::clamp(static_cast<std::int16_t>(value * max16), min16, max16)));
        }

        sound.stop();
        buffer.loadFromSamples(sampleValues.data(), (sf::Uint64)sampleValues.size(), 1, SMP_FREQ);
        sound.setBuffer(buffer);
    }
};

std::unique_ptr<Sample> PrepareGuitar()
{
    const std::string name = "KS Guitar";
    const float frequency = MIDDLE_C / 2.0f;
    const float decay = 0.005f;
    const float frequencyMultiplier = 1.0f;
    const float lengthInSeconds = 1.0f;
    const float filterAmount = 0.6f;
    const float filterInitial = 0.1f;
    const float filterFinal = 0.0004f;

    return std::make_unique<SampleKSSynth>(name, frequency, decay, filterAmount, lengthInSeconds, frequencyMultiplier, filterInitial, filterFinal);
}

std::unique_ptr<Sample> PrepareBass()
{
    const std::string name = "KS Bass";
    const float frequency = MIDDLE_C / 4.0f;
    const float decay = 0.005f;
    const float frequencyMultiplier = 0.5f;
    const float lengthInSeconds = 0.7f;
    const float filterAmount = 0.2f;
    const float filterInitial = 0.2f;
    const float filterFinal = 0.005f;

    return std::make_unique<SampleKSSynth>(name, frequency, decay, filterAmount, lengthInSeconds, frequencyMultiplier, filterInitial, filterFinal);
}

std::unique_ptr<Sample> PrepareKick()
{
    const std::string name = "Kick";

    return std::make_unique<SampleKicker>(name);
}

std::unique_ptr<Sample> PrepareHHClosed()
{
    const std::string name = "NH Hihat Closed";
    const std::uint8_t gvol = 32;
    const float decay = 0.03f;
    const float frequencyLow = 0.99f;
    const float frequencyHigh = 0.20f;

    return std::make_unique<SampleNoiseHit>(name, gvol, decay, frequencyLow, frequencyHigh);
}

std::unique_ptr<Sample> PrepareHHOpen()
{
    const std::string name = "NH Hihat Open";
    const std::uint8_t gvol = 32;
    const float decay = 0.5f;
    const float frequencyLow = 0.99f;
    const float frequencyHigh = 0.20f;

    return std::make_unique<SampleNoiseHit>(name, gvol, decay, frequencyLow, frequencyHigh);
}

std::unique_ptr<Sample> PrepareSnare()
{
    const std::string name = "NH Snare"; 
    const float decay = 0.12f;
    const float frequencyLow = 0.15f;
    const float frequencyHigh = 0.149f;

    return std::make_unique<SampleNoiseHit>(name, decay, frequencyLow, frequencyHigh);
}

const std::string GetDefaultSaveFolder()
{
    const std::string musicsFolder = "musics";
    if (std::filesystem::exists(musicsFolder) && std::filesystem::is_directory(musicsFolder))
        return musicsFolder + "/";
    else
        return "";
}

int main()
{
    srand(static_cast<unsigned>(time(nullptr)));

    const bool saveToITFile = false;
    const bool saveToOGGFile = true;
    const bool saveSamplesToWAVFile = false;

    NameGenerator nameGenerator;

    std::cout << "Generating samples...";
    std::array<SampleData, (std::size_t)Instruments::COUNT> samples;
    for (int i = 0; i < (int)Instruments::COUNT; ++i)
    {
        SampleData& sampleData = samples[i];

        switch ((Instruments)i)
        {
        case Instruments::Guitar: sampleData.sample = PrepareGuitar(); break;
        case Instruments::Bass: sampleData.sample = PrepareBass(); break;
        case Instruments::Kick: sampleData.sample = PrepareKick(); break;
        case Instruments::HHClosed: sampleData.sample = PrepareHHClosed(); break;
        case Instruments::HHOpen: sampleData.sample = PrepareHHOpen(); break;
        case Instruments::Snare: sampleData.sample = PrepareSnare(); break;
        }

        sampleData.Generate();

        if (saveSamplesToWAVFile)
        {
            sampleData.buffer.saveToFile(GetDefaultSaveFolder() + "sample-" + std::to_string(i) + ".wav");
        }
    }
    std::cout << " ==> DONE" << std::endl;

    std::cout << "Generating patterns...";
    int basenote = (std::rand() % (61 - 50 + 1) + 50) + 12;
    bool useMinorNote = (std::rand() % 100 <= 60) ? true : false;
    const int patternSize = 128;
    const int blockSize = 32;
    Strategy strat(useMinorNote ? static_cast<Key>(KeyMinor(basenote)) : static_cast<Key>(KeyMajor(basenote)), patternSize, blockSize);
    strat.AddGenerator<GeneratorBass>((int)Instruments::Bass);
    strat.AddGenerator<GeneratorDrums>((int)Instruments::Kick, (int)Instruments::HHClosed, (int)Instruments::HHOpen, (int)Instruments::Snare);
    strat.AddGenerator<GeneratorAmbientMelody>((int)Instruments::Guitar);
    std::vector<Pattern*> tempPatterns;
    const int patterns = 1;
    for (int i = 0; i < patterns; ++i)
    {
        tempPatterns.push_back(strat.MakePattern());
    }
    std::cout << " ==> DONE" << std::endl;

    int tempo = rand() % (160 - 60 + 1) + 60;
    std::cout << "Tempo: " << tempo << std::endl;

    std::cout << "Saving file..." << std::endl;

    // ITFile
    if (saveToITFile)
    {
        ITFile::Parameters parameters;

        parameters.tempo = tempo;

        parameters.samples.reserve(samples.size());
        for (auto& sampleData : samples)
            parameters.samples.push_back(sampleData.sample.get());

        parameters.patterns.reserve(tempPatterns.size());
        for (auto& patternPtr : tempPatterns)
            parameters.patterns.push_back(patternPtr);

        std::string name = nameGenerator.GenerateName();
        std::string filename = "bu-" + name;
        for (char& c : filename)
            if (c == ' ' || c == '\'')
                c = '-';
        filename += ".it";
        filename = GetDefaultSaveFolder() + filename;

        ITFile::Save(name, filename, parameters);
    }

    const int maxTracks = [&]()
    {
        std::set<std::size_t> tracks;
        for (Pattern* pattern : tempPatterns)
        {
            const std::vector<std::vector<Pattern::Chunk>>& chunks = pattern->GetData();
            for (std::size_t time = 0; time < chunks.size(); ++time)
            {
                for (std::size_t trackIndex = 0; trackIndex < chunks[time].size(); ++trackIndex)
                {
                    if (chunks[time][trackIndex].note != 253)
                    {
                        tracks.insert(trackIndex);
                    }
                }
            }
        }
        return (int)tracks.size();
    }();
    std::cout << "MaxTracks: " << maxTracks << std::endl;

    constexpr int debugTrack = 0; // debug

    const std::size_t notesPerUnit = blockSize * SMP_FREQ * 60 / (patternSize * tempo);
    std::cout << "NotesPerUnit: " << notesPerUnit << std::endl;
    std::cout << "UnitPerSeconds: " << (float)notesPerUnit / (float)SMP_FREQ << std::endl;

    constexpr sf::Int16 min16 = -0x8000;
    constexpr sf::Int16 max16 = 0x7FFF;

    sf::SoundBuffer musicBuffer;
    std::vector<sf::Int16> musicSamples;
    musicSamples.reserve(tempPatterns.size() * notesPerUnit * patternSize);
    std::cout << "ReservedSize: " << musicSamples.capacity() << std::endl;
    for (Pattern* pattern : tempPatterns)
    {
        const std::vector<std::vector<Pattern::Chunk>>& chunks = pattern->GetData();

        struct InstrumentInfo
        {
            std::uint8_t instrumentIndex = 0;
            Sample* instrument = nullptr;
            int dataIndex = 0;
            int dataLength = 0;
            float volume = 1.0f;
        };
        std::vector<InstrumentInfo> lastIntruments;
        lastIntruments.resize(64);

        for (std::size_t time = 0; time < chunks.size(); ++time)
        {
            for (std::size_t trackIndex = 0; trackIndex < chunks[time].size(); ++trackIndex)
            {
                if (trackIndex >= (std::size_t)maxTracks)
                    continue;

                const Pattern::Chunk& chunk = chunks[time][trackIndex];
                InstrumentInfo& lastInstrument = lastIntruments[trackIndex];

                if (debugTrack == (int)trackIndex)
                    std::cout << (int)chunk.note << " " << (int)chunk.instrument << " " << (int)chunk.volume << " " << (int)chunk.effectType << " " << (int)chunk.effectParameter << std::endl;

                if (chunk.note != 253) // New note
                {
                    lastInstrument.instrumentIndex = chunk.instrument;
                    lastInstrument.instrument = samples[chunk.instrument].sample.get();
                    lastInstrument.dataIndex = 0;
                    lastInstrument.dataLength = (int)lastInstrument.instrument->GetData().size();
                    lastInstrument.volume = ((int)chunk.volume) / 255.0f;

                    if (chunk.effectType == IT_EFFECT_SAMPLEOFFSET)
                    {
                        lastInstrument.dataIndex = chunk.effectParameter;
                    }
                }
            }

            for (std::size_t i = 0; i < notesPerUnit; ++i)
            {
                float unitMixedSample = 0.0f;
                int unitSampleCount = 0;

                for (std::size_t trackIndex = 0; trackIndex < chunks[time].size(); ++trackIndex)
                {
                    if (trackIndex >= (std::size_t)maxTracks)
                        continue;

                    const Pattern::Chunk& chunk = chunks[time][trackIndex];
                    InstrumentInfo& lastInstrument = lastIntruments[trackIndex];

                    if (lastInstrument.instrument != nullptr && lastInstrument.dataIndex < lastInstrument.dataLength)
                    {
                        unitSampleCount++;
                        unitMixedSample += lastInstrument.volume * lastInstrument.instrument->GetData()[lastInstrument.dataIndex];
                        lastInstrument.dataIndex++;
                    }
                }

                const float sampleFloat = unitSampleCount > 0 ? unitMixedSample / unitSampleCount : 0.0f;
                musicSamples.push_back(std::clamp(static_cast<sf::Int16>(sampleFloat* max16), min16, max16));
            }
        }
    }
    musicBuffer.loadFromSamples(musicSamples.data(), musicSamples.size(), 1, SMP_FREQ);
    std::cout << "FinalSize:    " << musicSamples.size() << std::endl;

    if (saveToOGGFile)
    {
        std::string filename = nameGenerator.GenerateName();
        for (char& c : filename)
            if (c == ' ' || c == '\'')
                c = '-';
        filename += ".ogg";

        musicBuffer.saveToFile(GetDefaultSaveFolder() + filename);
    }

    float samplesVolume = 10.0f;
    for (auto& sampleData : samples)
        sampleData.sound.setVolume(samplesVolume);

    sf::Sound music;
    music.setBuffer(musicBuffer);
    music.setVolume(samplesVolume);

    std::string lastGeneratedName = nameGenerator.GenerateName();

    Application app;
    app.Run([&]()
    {  
        if (ImGui::Begin("Autotracker"))
        {
            if (ImGui::CollapsingHeader("NameGenerator"))
            {
                ImGui::Indent();

                ImGui::Text("%s", lastGeneratedName.c_str());
                if (ImGui::Button("Generate new name"))
                {
                    lastGeneratedName = nameGenerator.GenerateName();
                }

                ImGui::Unindent();
            }
            if (ImGui::CollapsingHeader("Samples"))
            {
                ImGui::Indent();
                if (ImGui::Button("Regenerate"))
                {
                    for (int i = 0; i < (int)Instruments::COUNT; ++i)
                    {
                        samples[i].Generate();
                    }
                }

                if (ImGui::SliderFloat("SamplesVolume", &samplesVolume, 0.0f, 100.0f))
                {
                    for (auto& sampleData : samples)
                        sampleData.sound.setVolume(samplesVolume);
                }

                for (int i = 0; i < (int)Instruments::COUNT; ++i)
                {
                    SampleData& sampleData = samples[i];
                    ImGui::PushID(i);
                    sampleData.DisplayImGui();
                    ImGui::PopID();
                }

                ImGui::Unindent();
            }
            if (ImGui::CollapsingHeader("Music"))
            {
                ImGui::Indent();

                if (ImGui::Button("Play"))
                    music.play();

                if (ImGui::SliderInt("Tempo", &tempo, 60, 160))
                    std::cout << tempo << std::endl;

                ImGui::Unindent();
            }
        }
        ImGui::End();
    });

    return 0;
}