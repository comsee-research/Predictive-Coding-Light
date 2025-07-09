//
// Created by thomas on 07/07/22.
//

#include "config/DefaultConfig.hpp"

namespace PredefinedConfigurations {
    NetConf twoLayerOnePatchWeightSharingCenteredConfig() {
        NetConf conf = {
                {
                        {"nbCameras",  1},
                        {"neuron1Synapses", 1},
                        {"sharingType", "patch"},
                        {"neuronType",  {"SimpleCell", "ComplexCell"}},
                        {"layerInhibitions",  {{"local"}, {"local"}}},
                        {"interLayerConnections", {{-1}, {0}}},
                        {"layerPatches",  {{{150}, {100}, {0}}, {{0}, {0}, {0}}}},
                        {"layerSizes",    {{9, 9, 64}, {6, 6, 32}}},
                        {"neuronSizes", {{{10, 10, 1}}, {{4, 4, 64}}}},
                        {"neuronOverlap",     {{3, 3, 1}, {3, 3, 64}}},
                        {"neuronInhibitionRange", {4, 4}},
                        {"vfWidth",       346},
                        {"vfHeight",    260},
                        {"measurementInterval", 100}
                },
                {
                        {"VTHRESH",    30},
                        {"VRESET",          -10},
                        {"TRACKING",    "none"},
                        {"POTENTIAL_TRACK", {4,            4}},
                        {"TAU_SRA",           100},
                        {"TAU_RP",                5},
                        {"TAU_M",         18},
                        {"TAU_LTP",       7},
                        {"TAU_LTD",     7},
                        {"TARGET_SPIKE_RATE", 0.75},
                        {"SYNAPSE_DELAY",         0},
                        {"STDP_LEARNING", "all"},
                        {"NORM_FACTOR", 50},
                        {"LATERAL_NORM_FACTOR", 6500},
                        {"TOPDOWN_NORM_FACTOR", 2000},
                        {"DECAY_RATE", 0},
                        {"MIN_THRESH", 4},
                        {"ETA_LTP", 0.00077},
                        {"ETA_LTD", -0.00021},
                        {"ETA_ILTP", 0.0004600},
                        {"ETA_ILTD", -0.0004600},
                        {"ETA_SRA", 0},
                        {"ETA_TA", 0},
                        {"ETA_RP", 10},
                        {"ETA_INH", 1500},
                },
                {
                        {"VTHRESH",    3},
                        {"VRESET",          -10},
                        {"TRACKING",    "none"},
                        {"POTENTIAL_TRACK", {2,            3}},
                        {"TAU_M",           50},
                        {"TAU_LTP",           40},
                        {"TAU_LTD",               40},
                        {"TAU_RP",        5},
                        {"STDP_LEARNING", "all"},
                        {"NORM_FACTOR", 1000},
                        {"DECAY_RATE",        0},
                        {"ETA_LTP",               0.002},
                        {"ETA_LTD",       0.002},
                        {"ETA_ILTP", 0.008},
                        {"ETA_ILTD", -0.008},
                        {"ETA_INH",     600},
                        {"ETA_RP",              10},
                }
        };
        return conf;
    }
}