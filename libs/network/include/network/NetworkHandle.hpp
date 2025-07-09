//
// Created by Thomas on 06/05/2021.
//

#ifndef NEUVISYS_DV_NETWORK_HANDLE_HPP
#define NEUVISYS_DV_NETWORK_HANDLE_HPP

#include "SpikingNetwork.hpp"

struct H5EventFile {
    H5::H5File file;
    H5::Group group;
    H5::DataSet timestamps;
    H5::DataSet x;
    H5::DataSet y;
    H5::DataSet polarities;
    H5::DataSet cameras;
    hsize_t dims;
    hsize_t packetSize = 10000;
    hsize_t offset = 0;
    hsize_t countPass = 0;
    uint64_t firstTimestamp = 0;
    uint64_t lastTimestamp = 0;
};

struct SaveTime {
    double action{};
    double update{};
    double console{};
    double display{};

    explicit SaveTime(double initTime) : action(initTime), update(initTime), console(initTime), display(initTime) {};
};

/**
 * Used as an abstraction layer on top of the SpikingNetwork class.
 * It offers functions used for communication between the environment (mainly the incoming flow of events) and the spiking neural network.
 * Example:
 *      network = NetworkHandle("/path/to/network/");
 *      network.transmitEvents(eventPacket);
 */
class NetworkHandle {
    SpikingNetwork m_spinet;
    NetworkConfig m_networkConf;
    NeuronConfig m_simpleNeuronConf;
    NeuronConfig m_complexNeuronConf;
    H5EventFile m_eventFile;
    SaveTime m_saveTime;

    std::map<std::string, std::vector<double>> m_saveData;
    double m_reward{};
    double m_neuromodulator{};
    std::string m_eventsPath;
    size_t m_nbEvents{};
    int m_action{};
    size_t m_iteration{};
    size_t m_packetCount{};
    size_t m_actionCount{};
    size_t m_scoreCount{};
    size_t m_countEvents{};
    size_t m_totalNbEvents{};
    size_t m_saveCount{};
    double m_endTime{};
    double m_averageEventRate{};
public:
    NetworkHandle();

    explicit NetworkHandle(const std::string& eventsPath, double time);

    explicit NetworkHandle(const std::string &networkPath);

    explicit NetworkHandle(const std::string &networkPath, const std::string &eventsPath);

    void setEventPath(const std::string &eventsPath);

    bool loadEvents(std::vector<Event> &events, size_t nbPass);

    void feedEvents(const std::vector<Event> &events);

    void transmitEvent(const Event &event);

    void learningDecay(double time);

    void intermediateSave(size_t nbRun);

    void save(const std::string &eventFileName, size_t nbRun);

    void saveStatistics(size_t simulation, size_t sequence, const std::string& folderName, bool reset=false, bool sep_speed = false, int n_speed = 2);

    void trackNeuron(long time, size_t id = 0, size_t layer = 0);

    void updateNeurons(size_t time);

    std::map<std::string, std::vector<double>> &getSaveData() { return m_saveData; }

    std::reference_wrapper<Neuron> &getNeuron(size_t index, size_t layer) { return m_spinet.getNeuron(index, layer); }

    const std::vector<size_t> &getNetworkStructure() { return m_spinet.getNetworkStructure(); }

    uint64_t getLayout(size_t layer, Position pos) { return m_spinet.getLayout()[layer][{pos.x(), pos.y(), pos.z()}]; }

    cv::Mat neuronWeightMatrix(size_t idNeuron, size_t layer, size_t camera, size_t synapse, size_t z);

    cv::Mat getSummedWeightNeuron(size_t idNeuron, size_t layer);

    NetworkConfig getNetworkConfig() { return m_networkConf; }

    NeuronConfig getSimpleNeuronConfig() { return m_simpleNeuronConf; }

    NeuronConfig getComplexNeuronConfig() { return m_complexNeuronConf; }

    [[nodiscard]] uint64_t getFirstTimestamp() const { return m_eventFile.firstTimestamp; }

    [[nodiscard]] uint64_t getLastTimestamp() const { return m_eventFile.lastTimestamp; }

    void changeNeuronToTrack(int n_x, int n_y);

    void lateralRandom(int norm_factor);

    void inhibitionShuffle(int case_);

    void assignOrientation(int z, int ori, int thickness = 1);

    void assignComplexOrientation(int neur, int ori, int thickness = 1);

    void setSequenceParameters(std::vector<std::vector<size_t>> parameters);

    void deactivateDynamicInhib(bool activation);

    void resetAllNeurons();

    void normalizeL1Weights();

    void assignPatchSize(double patch);


private:
    void load();

    void loadNpzEvents(std::vector<Event> &events, size_t nbPass = 1);

    void openH5File();

    bool loadHDF5Events(std::vector<Event> &events, size_t nbPass);

    void readFirstAndLastTimestamp();

};

#endif //NEUVISYS_DV_NETWORK_HANDLE_HPP
