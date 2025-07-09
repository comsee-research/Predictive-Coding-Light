//
// Created by thomas on 06/07/22.
//

#ifndef NEUVISYS_SPIKINGNETWORKTEST_HPP
#define NEUVISYS_SPIKINGNETWORKTEST_HPP

#include "gtest/gtest.h"

#include "network/SpikingNetwork.hpp"

class SpikingNetworkTest : public ::testing::Test {
protected:
    void SetUp() override;

    void TearDown() override;

    SpikingNetwork spinet;
};


#endif //NEUVISYS_SPIKINGNETWORKTEST_HPP
