//
// Created by thomas on 06/07/22.
//

#include "SpikingNetworkTest.hpp"

void SpikingNetworkTest::SetUp() {
    NetworkConfig::createNetwork("../../data/networks/network_test", PredefinedConfigurations::twoLayerOnePatchWeightSharingCenteredConfig);
    spinet = SpikingNetwork("../../data/networks/network_test/");
}

void SpikingNetworkTest::TearDown() {
    std::filesystem::remove_all("../../data/networks/network_test/");
}

TEST_F(SpikingNetworkTest, folderCreation) {
    EXPECT_EQ(std::filesystem::exists("../../data/networks/network_test"), true);
}

TEST_F(SpikingNetworkTest, networkCreation) {
    EXPECT_EQ(spinet.getNetworkStructure()[0], 9*9*64);
    EXPECT_EQ(spinet.getNetworkStructure()[1], 6*6*32);
}