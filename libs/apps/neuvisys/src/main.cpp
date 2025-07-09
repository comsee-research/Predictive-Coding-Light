//
// Created by Thomas on 04/06/2021.
//

#include <network/NetworkHandle.hpp>
#include <network/SurroundSuppression.hpp>
#include <network/config/DefaultConfig.hpp>

int main(int argc, char *argv[]) {
    if (argc > 2) {
        std::string networkPath = argv[1];
        std::string eventsPath = argv[2];
        NetworkHandle network(networkPath, eventsPath);
        size_t nbCount = std::atoi(argv[3]);
        std::vector<Event> events;
        std::cout << "argv[3] = " << nbCount << std::endl;
        std::cout << "Feeding network... " << std::endl;

        while (network.loadEvents(events, nbCount)) {
            network.feedEvents(events);
        }
        network.save(eventsPath, nbCount);
    } else if (argc > 1) {
        NetworkConfig::createNetwork(argv[1], PredefinedConfigurations::twoLayerOnePatchWeightSharingCenteredConfig);
    } else {

        std::cout << "too few arguments, entering debug mode" << std::endl;
        std::string networkPath = "/home/comsee/PhD_Antony/data_basic_PCL_NatComms/net1c/";
    
        std::string path_Events = "/home/comsee/PhD_Antony/data_basic_PCL_NatComms/test/";

        std::vector<std::string> vectorOfPaths;
        for (const auto & frame : std::filesystem::directory_iterator{path_Events}) {
            vectorOfPaths.emplace_back(frame.path().string());
        }
        NetworkHandle network(networkPath, vectorOfPaths[0]);
        SurroundSuppression surround(networkPath,vectorOfPaths,network);
        surround.recordSpikes(path_Events);
    }
}
