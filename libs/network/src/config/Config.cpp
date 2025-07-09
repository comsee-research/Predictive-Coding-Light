//
// Created by Thomas on 14/04/2021.
//

#include "Config.hpp"

NetworkConfig::NetworkConfig() = default;

NetworkConfig::NetworkConfig(const std::string &configFile) {
    m_networkConfigPath = configFile;
    std::string toRemove = "configs/network_config.json";
    m_networkPath = m_networkConfigPath.substr(0, m_networkConfigPath.size() - toRemove.size());
    loadNetworkLayout();
}

void NetworkConfig::loadNetworkLayout() {
    nlohmann::json conf;

    std::ifstream ifs(m_networkConfigPath);
    std::cout << m_networkConfigPath << std::endl;
    if (ifs.is_open()) {
        try {
            ifs >> conf;
            nbCameras = conf["nbCameras"];
            neuron1Synapses = conf["neuron1Synapses"];
            sharingType = conf["sharingType"];
            vfWidth = conf["vfWidth"];
            vfHeight = conf["vfHeight"];
            measurementInterval = E3 * static_cast<double>(conf["measurementInterval"]);
            neuronInhibitionRange = static_cast<std::vector<size_t>>(conf["neuronInhibitionRange"]);

            for (size_t i = 0; i < conf["neuronType"].size(); ++i) {
                connections.emplace_back();
                connections[i].neuronType = static_cast<std::string>(conf["neuronType"][i]);
                connections[i].inhibitions = static_cast<std::vector<std::string>>(conf["layerInhibitions"][i]);
                connections[i].interConnections = static_cast<std::vector<int>>(conf["interLayerConnections"][i]);
                connections[i].patches = static_cast<std::vector<std::vector<size_t>>>(conf["layerPatches"][i]);
                connections[i].sizes = static_cast<std::vector<size_t>>(conf["layerSizes"][i]);
                connections[i].neuronSizes = static_cast<std::vector<std::vector<size_t>>>(conf["neuronSizes"][i]);
                connections[i].neuronOverlap = static_cast<std::vector<size_t>>(conf["neuronOverlap"][i]);
            }
        } catch (const std::exception &e) {
            std::cerr << "In network config file:" << e.what() << std::endl;
            throw;
        }
    } else {
        std::cout << "cannot open network configuration file" << std::endl;
    }
    ifs.close();
}

NeuronConfig::NeuronConfig() = default;

NeuronConfig::NeuronConfig(const std::string &configFile, size_t type) {
    if (type == 0) {
        loadSimpleNeuronsParameters(configFile);
    } else if (type == 1) {
        loadComplexNeuronsParameters(configFile);
    } 
}

void NeuronConfig::loadSimpleNeuronsParameters(const std::string &fileName) {
    nlohmann::json conf;

    std::ifstream ifs(fileName);
    if (ifs.is_open()) {
        try {
            ifs >> conf;
            ETA_LTP = conf["ETA_LTP"];
            ETA_LTD = conf["ETA_LTD"];
            ETA_ILTP = conf["ETA_ILTP"];
            ETA_ILTD = conf["ETA_ILTD"];
            ETA_SR = conf["ETA_TA"];
            DELTA_RP = conf["ETA_RP"];
            DELTA_SRA = conf["ETA_SRA"];
            ETA_INH = conf["ETA_INH"];
            TAU_LTP = E3 * static_cast<double>(conf["TAU_LTP"]);
            TAU_LTD = E3 * static_cast<double>(conf["TAU_LTD"]);
            TAU_RP = E3 * static_cast<double>(conf["TAU_RP"]);
            TAU_M = E3 * static_cast<double>(conf["TAU_M"]);
            TAU_SRA = E3 * static_cast<double>(conf["TAU_SRA"]);
            VTHRESH = conf["VTHRESH"];
            VRESET = conf["VRESET"];
            SYNAPSE_DELAY = conf["SYNAPSE_DELAY"];
            NORM_FACTOR = conf["NORM_FACTOR"];
            LATERAL_NORM_FACTOR = conf["LATERAL_NORM_FACTOR"];
            TOPDOWN_NORM_FACTOR = conf["TOPDOWN_NORM_FACTOR"];
            DECAY_RATE = conf["DECAY_RATE"];
            TARGET_SPIKE_RATE = conf["TARGET_SPIKE_RATE"];
            MIN_THRESH = conf["MIN_THRESH"];
            STDP_LEARNING = conf["STDP_LEARNING"];
            TRACKING = conf["TRACKING"];
            POTENTIAL_TRACK = static_cast<std::vector<int>>(conf["POTENTIAL_TRACK"]);
        } catch (const std::exception &e) {
            std::cerr << "In simple cell config file" << std::endl;
            throw;
        }
    } else {
        std::cout << "cannot open neuron configuration file" << std::endl;
    }
    ifs.close();
}

void NeuronConfig::loadComplexNeuronsParameters(const std::string &fileName) {
    nlohmann::json conf;

    std::ifstream ifs(fileName);
    if (ifs.is_open()) {
        try {
            ifs >> conf;
            ETA_LTP = conf["ETA_LTP"];
            ETA_LTD = conf["ETA_LTD"];
            ETA_ILTP = conf["ETA_ILTP"];
            ETA_ILTD = conf["ETA_ILTD"];
            TAU_LTP = E3 * static_cast<double>(conf["TAU_LTP"]);
            TAU_LTD = E3 * static_cast<double>(conf["TAU_LTD"]);
            TAU_M = E3 * static_cast<double>(conf["TAU_M"]);
            TAU_RP = E3 * static_cast<double>(conf["TAU_RP"]);
            VTHRESH = conf["VTHRESH"];
            ETA_INH = conf["ETA_INH"];
            VRESET = conf["VRESET"];
            NORM_FACTOR = conf["NORM_FACTOR"];
            DECAY_RATE = conf["DECAY_RATE"];
            STDP_LEARNING = conf["STDP_LEARNING"];
            TRACKING = conf["TRACKING"];
            DELTA_RP = conf["ETA_RP"];
            POTENTIAL_TRACK = static_cast<std::vector<int>>(conf["POTENTIAL_TRACK"]);
        } catch (const std::exception &e) {
            std::cerr << "In complex cell config file" << std::endl;
            throw;
        }
    } else {
        std::cout << "cannot open neuron configuration file" << std::endl;
    }
    ifs.close();
}

void NetworkConfig::createNetworkDirectory(const std::string &directory) {
    fs::create_directory(directory);
    fs::create_directory(directory + "/configs");
    fs::create_directory(directory + "/figures");
    fs::create_directory(directory + "/figures/0");
    fs::create_directory(directory + "/figures/1");
    fs::create_directory(directory + "/gabors");
    fs::create_directory(directory + "/gabors/0");
    fs::create_directory(directory + "/gabors/1");
    fs::create_directory(directory + "/gabors/figures");
    fs::create_directory(directory + "/gabors/hists");
    fs::create_directory(directory + "/images");
    fs::create_directory(directory + "/images/0");
    fs::create_directory(directory + "/images/1");
    fs::create_directory(directory + "/weights");
    fs::create_directory(directory + "/weights/0");
    fs::create_directory(directory + "/weights/1");
    fs::create_directory(directory + "/statistics");
    fs::create_directory(directory + "/statistics/0");
    fs::create_directory(directory + "/statistics/1");
}

void NetworkConfig::createNetwork(const std::string &directory, const std::function<NetConf()> &config) {
    createNetworkDirectory(directory);
    auto conf = config();
    size_t count = 0;
    for (auto file: {"configs/network_config.json", "configs/simple_cell_config.json", "configs/complex_cell_config.json"}) {
        std::ofstream ofs(directory + "/" + file);
        if (ofs.is_open()) {
            ofs << std::setw(4) << conf[count] << std::endl;
        } else {
            std::cout << "cannot create network files" << std::endl;
        }
        ofs.close();
        ++count;
    }
}
