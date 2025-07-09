//
// Created by Thomas on 14/04/2021.
//

#ifndef NEUVISYS_DV_CONFIG_HPP
#define NEUVISYS_DV_CONFIG_HPP

#include <iostream>
#include <fstream>
#include <iomanip>
#include <utility>

#if __GNUC__ > 8
	#include <filesystem>
	namespace fs = std::filesystem;
#else
	#include <experimental/filesystem> //if gcc < 8
	namespace fs = std::experimental::filesystem;
#endif

#include "json/json.hpp"

#include "DefaultConfig.hpp"

static constexpr std::size_t E3 = 1000; // µs
static constexpr std::size_t E6 = 1000000; // µs


struct LayerConnectivity {
    std::string neuronType;
    std::vector<std::string> inhibitions;
    std::vector<int> interConnections;
    std::vector<std::vector<size_t>> patches;
    std::vector<size_t> sizes;
    std::vector<std::vector<size_t>> neuronSizes;
    std::vector<size_t> neuronOverlap;
};

class NetworkConfig {
    /***** Display parameters *****/
    std::string m_networkPath;
    std::string m_networkConfigPath;

    /***** Spiking Neural Network layout parameters *****/
    size_t nbCameras{};
    size_t neuron1Synapses{};
    std::string sharingType{};
    size_t vfWidth{};
    size_t vfHeight{};
    double measurementInterval{};
    std::vector<size_t> neuronInhibitionRange;
    std::vector<LayerConnectivity> connections;

public:
    NetworkConfig();

    explicit NetworkConfig(const std::string& configFile);

    void loadNetworkLayout();

    std::string &getNetworkPath() { return m_networkPath; }

    std::string &getNetworkConfigPath() { return m_networkConfigPath; }

    std::string &getSharingType() { return sharingType; }

    [[nodiscard]] size_t getNbCameras() const { return nbCameras; }

    [[nodiscard]] size_t getNeuron1Synapses() const { return neuron1Synapses; }

    [[nodiscard]] size_t getVfWidth() const { return vfWidth; }

    [[nodiscard]] size_t getVfHeight() const { return vfHeight; }

    [[nodiscard]] double getMeasurementInterval() const { return measurementInterval; }

    std::vector<LayerConnectivity> &getLayerConnectivity() { return connections; }

    std::vector<size_t> getNeuronInhibitionRange() { return neuronInhibitionRange; }

    static void createNetwork(const std::string &directory, const std::function<NetConf()> &config);

private:
    static void createNetworkDirectory(const std::string &directory);

};

class NeuronConfig {
public:
    NeuronConfig();

    NeuronConfig(const std::string &configFile, size_t type);

/***** Neurons internal parameters *****/
    double TAU_M{}; // μs
    double TAU_LTP{}; // μs
    double TAU_LTD{}; // μs
    double TAU_RP{}; // μs
    double TAU_SRA{}; // μs
    double TAU_E{}; // μs
    double TAU_K{}; // μs
    double NU_K{}; // μs
    double MIN_NU_K{}; // μs
    double MIN_TAU_K{}; // μs

    double ETA_LTP{}; // mV
    double ETA_LTD{}; // mV
    double ETA_ILTP{}; // mV
    double ETA_ILTD{}; // mV
    double ETA_SR{}; // mV
    double DELTA_RP{}; // mv
    double DELTA_SRA{}; // mV
    double ETA_INH{}; // mV
    double ETA{}; // mV

    double VRESET{}; // mV
    double VTHRESH{}; // mV

    size_t SYNAPSE_DELAY{}; // μs

    double NORM_FACTOR{};
    double LATERAL_NORM_FACTOR{};
    double TOPDOWN_NORM_FACTOR{};
    double DECAY_RATE{};

    double TARGET_SPIKE_RATE{}; // spikes/s
    double MIN_THRESH{}; // mV

    std::string STDP_LEARNING{};
    std::string TRACKING{};
    std::vector<int> POTENTIAL_TRACK;

private:
    void loadSimpleNeuronsParameters(const std::string &fileName);

    void loadComplexNeuronsParameters(const std::string &fileName);
};

#endif //NEUVISYS_DV_CONFIG_HPP
