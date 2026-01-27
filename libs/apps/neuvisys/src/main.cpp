//
// Created by Thomas on 04/06/2021.
//

#include <network/NetworkHandle.hpp>
#include <network/SurroundSuppression.hpp>
#include <network/config/DefaultConfig.hpp>

int main(int argc, char *argv[]) {
    bool recordSpikes = false;

    if (argc == 5){
        std::cout << "Spike recording enabled." << std::endl;
        std::string recordFlag = argv[4];
        std::istringstream(recordFlag) >> std::boolalpha >> recordSpikes;
        std::cout << std::boolalpha;
        std::cout << "recordSpikes = " << recordSpikes << std::endl;
        std::cout << std::noboolalpha;
    }

    if (argc > 2) {
        std::string networkPath = argv[1];
        std::string eventsPath = argv[2];
        NetworkHandle network(networkPath, eventsPath);
        size_t nbCount = std::atoi(argv[3]);
        std::vector<Event> events;
        std::cout << "argv[3] = " << nbCount << std::endl;
        std::cout << "Feeding network... " << std::endl;

        // ===== SPIKE RECORDING FOR CLASSIFICATION =====
        // Uncomment and configure this section to record spikes for DVS-Gesture128 classification
        // You need to set the paths to your 11 gesture folders and provide the corresponding labels
        if (recordSpikes) {
            std::cout << "\n===== Recording spikes for classification =====" << std::endl;
            
            // Initialize gesture paths - update these paths to match your DVS-Gesture128 folder structure
            std::vector<std::string> gesturePaths = {
                eventsPath + "/0_hand_clapping",
                eventsPath + "/1_right_hand_wave",
                eventsPath + "/2_left_hand_wave",
                eventsPath + "/3_right_arm_clockwise",
                eventsPath + "/4_right_arm_counter_clockwise",
                eventsPath + "/5_left_arm_clockwise",
                eventsPath + "/6_left_arm_counter_clockwise",
                eventsPath + "/7_arm_roll",
                eventsPath + "/8_air_drums",
                eventsPath + "/9_air_guitar",
                eventsPath + "/10_other_gestures"
            };
            
            // Initialize gesture labels (0-10 for 11 gesture classes, label-1 is used in ClassificationDescriptor)
            std::vector<int> gestureLabels = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
            
            // Create SurroundSuppression object and record spikes
            SurroundSuppression surround(networkPath, gesturePaths, network);
            surround.classificationDescriptor(gesturePaths, gestureLabels);
            
            std::cout << "\n✓ Spike recording completed!" << std::endl;
            std::cout << "Output saved to: " << networkPath << "statistics/gesture/" << std::endl;
        } 
        // ===== NORMAL TRAINING/INFERENCE MODE =====
        else {
            std::cout << "Feeding network... " << std::endl;
            
            // Initialize network with event file
            NetworkHandle network(networkPath, eventsPath);
            
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
