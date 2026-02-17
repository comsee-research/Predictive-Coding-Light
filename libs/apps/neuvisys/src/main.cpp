//
// Created by Thomas on 04/06/2021.
//

#include <network/NetworkHandle.hpp>
#include <network/SurroundSuppression.hpp>
#include <network/config/DefaultConfig.hpp>
#include <network/DatasetScanner.hpp>

int main(int argc, char *argv[]) { 

    if (argc > 2) {
        std::string networkPath = argv[1];
        std::string eventsPath = argv[2];
        NetworkHandle network(networkPath, eventsPath);
        size_t nbCount = std::atoi(argv[3]);
        std::vector<Event> events;
        std::cout << "argv[3] = " << nbCount << std::endl;

        // Check if spike recording is enabled via config
        bool spikeRecordingEnabled = network.getNetworkConfig().isSpikeRecordingEnabled();

        if (spikeRecordingEnabled) {
            std::cout << "\n===== Spike Recording Mode =====" << std::endl;
            
            // Check if it's a labeled dataset (has class subdirectories)
            if (DatasetScanner::isLabeledDataset(eventsPath)) {
                // Use the new config-driven approach with auto-detection for datasets
                std::cout << "Detected labeled dataset" << std::endl;
                if (!network.recordSpikesForDataset(eventsPath)) {
                    std::cerr << "Spike recording failed!" << std::endl;
                    return 1;
                }
            } else {
                // Single file spike recording - DISABLED FOR NOW
                // TODO: Implement spike timing recording per neuron per layer
                // Currently, spike recording is only supported for labeled datasets (multi-file)
                // For single files, we need a different approach:
                //   - Option 1: Split into time windows and treat each as a sample
                //   - Option 2: Record detailed spike timing for each neuron (different format)
                //   - Option 3: Process entire file as one sample (current approach, commented out)
                
                std::cerr << "Error: Spike recording is only supported for labeled datasets." << std::endl;
                std::cerr << "Expected structure: " << eventsPath << "/<class_0>/*.npz, <class_1>/*.npz, ..." << std::endl;
                std::cerr << "For single-file spike analysis, please use normal inference mode." << std::endl;
                return 1;
                
                // std::cout << "Processing single file with spike recording" << std::endl;
                // const auto& srConfig = network.getNetworkConfig().getSpikeRecordingConfig();
                // std::cout << "Output folder: " << networkPath << "statistics/" << srConfig.outputSubfolder << "/" << std::endl;
                // 
                // while (network.loadEvents(events, nbCount)) {
                //     network.feedEvents(events);
                // }
                // 
                // // Save statistics for single file
                // // Folder structure: statistics/<outputFolder>/0/
                // std::string folderPath = srConfig.outputSubfolder;
                // network.saveStatistics(0, 1, folderPath, true);
                // 
                // std::cout << "\n✓ Spike recording completed!" << std::endl;
                // std::cout << "Output saved to: " << networkPath << "statistics/" << folderPath << std::endl;
            }
        } 
        // Normal training/inference mode (no spike recording)
        else {
            std::cout << "Feeding network... " << std::endl;
            
            while (network.loadEvents(events, nbCount)) {
                network.feedEvents(events);
            }
            network.save(eventsPath, nbCount);
        }
    } else if (argc > 1) {
        NetworkConfig::createNetwork(argv[1], PredefinedConfigurations::twoLayerOnePatchWeightSharingCenteredConfig);
    } else {

        std::cout << "too few arguments, entering debug mode" << std::endl;
        std::string networkPath = "/home/comsee/PhD_Antony/data_basic_PCL_NatComms/net1c/";
    
        std::string path_Events = "/home/comsee/PhD_Antony/data_basic_PCL_NatComms/test/";

        std::vector<std::string> vectorOfPaths;
        for (const auto & frame : std::experimental::filesystem::directory_iterator{path_Events}) {
            vectorOfPaths.emplace_back(frame.path().string());
        }
        NetworkHandle network(networkPath, vectorOfPaths[0]);
        SurroundSuppression surround(networkPath,vectorOfPaths,network);
        surround.recordSpikes(path_Events);
    }
}
