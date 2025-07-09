//
// Created by thomas on 06/07/22.
//

#include "NetworkHandleTest.hpp"

NetworkHandle* NetworkHandleTest::network = nullptr;
std::string NetworkHandleTest::eventsPath;
std::string NetworkHandleTest::networkPath;

void NetworkHandleTest::SetUpTestSuite() {
    eventsPath = "../../data/events/shapes.h5";
    EXPECT_EQ(std::filesystem::exists("../../data/events/shapes.h5"), true);
    networkPath = "../../data/networks/network_test/";

    NetworkConfig::createNetwork("../../data/networks/network_test", PredefinedConfigurations::twoLayerOnePatchWeightSharingCenteredConfig);
    if (network == nullptr) {
        WeightMatrix::setSeed(1486546);
        WeightMap::setSeed(461846);
        network = new NetworkHandle(networkPath, eventsPath);
    }
}

void NetworkHandleTest::TearDownTestSuite() {
    delete network;
    network = nullptr;
    std::filesystem::remove_all(networkPath);
}

TEST_F(NetworkHandleTest, runningNetwork) {
    while (network->loadEvents(events, 1)) {
        network->feedEvents(events);
    }
}

TEST_F(NetworkHandleTest, checkWeights) {

}

TEST_F(NetworkHandleTest, savingNetwork) {
    network->save(eventsPath, 1);
}

TEST_F(NetworkHandleTest, runningNetworkWeightSharing) {
    // network = new NetworkHandle(networkPath2, eventsPath);

    // while (network->loadEvents(events, 1)) {
    //     network->feedEvents(events);
    // }
}

TEST_F(NetworkHandleTest, savingNetworkWeightSharing) {
    // network->save(eventsPath, 1);
}