//
// Created by Thomas on 06/05/2021.
//

#include "NetworkHandle.hpp"

/**
 * Constructs an empty network.
 */
NetworkHandle::NetworkHandle() : m_saveTime(0) {

}

/**
 * Constucts the NetworkHandle without creating a SpikingNetwork object.
 * Used for reading event files.
 * @param eventsPath - Path to the event file.
 * @param time - Starting time for the network.
 */
NetworkHandle::NetworkHandle(const std::string &eventsPath, double time) : m_saveTime(time) {
    setEventPath(eventsPath);
}

/**
 * Constucts the NetworkHandle and creates a SpikingNetwork object associated to it.
 * Loads the configuration files locally.
 * Loads network weights if they exist.
 * @param networkPath - Path to the network folder.
 */
NetworkHandle::NetworkHandle(const std::string &networkPath) : m_spinet(networkPath),
                                                               m_networkConf(NetworkConfig(networkPath + "configs/network_config.json")),
                                                               m_simpleNeuronConf(m_networkConf.getNetworkPath() + "configs/simple_cell_config.json",
                                                                                  0),
                                                               m_complexNeuronConf(
                                                                       m_networkConf.getNetworkPath() + "configs/complex_cell_config.json", 1),
                                                               m_saveTime(0) {
    load();
}

/**
 * Overload NetworkHandle constructor.
 * Loads an event file in addition to the usual constructor.
 * @param networkPath - Path to the network folder.
 * @param eventsPath - Path to the event file.
 */
NetworkHandle::NetworkHandle(const std::string &networkPath,
                             const std::string &eventsPath) : NetworkHandle(networkPath) {
    setEventPath(eventsPath);
}

void NetworkHandle::setEventPath(const std::string &eventsPath) {
    m_eventsPath = eventsPath;

    std::string hdf5 = ".h5";
    if (std::equal(hdf5.rbegin(), hdf5.rend(), m_eventsPath.rbegin())) {
        openH5File();
    }
}

/**
 * Open an HDF5 event file.
 */
void NetworkHandle::openH5File() {
    if (!m_eventsPath.empty()) {
        m_eventFile.file = H5::H5File(m_eventsPath, H5F_ACC_RDONLY);
        m_eventFile.group = m_eventFile.file.openGroup("events");
        m_eventFile.timestamps = m_eventFile.group.openDataSet("./t");
        m_eventFile.x = m_eventFile.group.openDataSet("./x");
        m_eventFile.y = m_eventFile.group.openDataSet("./y");
        m_eventFile.polarities = m_eventFile.group.openDataSet("./p");
        m_eventFile.cameras = m_eventFile.group.openDataSet("./c");
        m_eventFile.timestamps.getSpace().getSimpleExtentDims(&m_eventFile.dims);
        m_nbEvents += m_eventFile.dims;
        readFirstAndLastTimestamp();
    }
}

/**
 * Load packets of events from an event file.
 * Two format are recognized.
 * HDF5: loads events packet by packet.
 * NPZ: load all the events at the same time.
 * @param events - vector that will be filled with the events from the file.
 * @param nbPass - number of repetition of the event file.
 * @return true if all the events have been loaded. false otherwise.
 */
bool NetworkHandle::loadEvents(std::vector<Event> &events, size_t nbPass) {
    m_endTime = static_cast<double>(nbPass) * (getLastTimestamp() - getFirstTimestamp());
    std::string hdf5 = ".h5";
    std::string npz = ".npz";
    if (std::equal(hdf5.rbegin(), hdf5.rend(), m_eventsPath.rbegin())) {
        if (loadHDF5Events(events, nbPass)) {
            return false;
        }
        return true;
    } else if (std::equal(npz.rbegin(), npz.rend(), m_eventsPath.rbegin())) {
        if (m_iteration > 0) {
            return false;
        }
        loadNpzEvents(events, nbPass);
        return true;
    }
}

/**
 * Feed a packet of events to the network.
 * @param events - Vector of events.
 */
void NetworkHandle::feedEvents(const std::vector<Event> &events) {
    m_nbEvents = events.size();
    m_endTime = (events.at(m_nbEvents-1).timestamp());// - events.at(0).timestamp());
    int counter = 0;
    for (const auto &event: events) {
        ++counter;
        ++m_iteration;
            if(counter == m_nbEvents) {
                const auto ev = Event(event.timestamp(), event.x(), event.y(), event.polarity(), event.camera(), event.synapse(), true);
                transmitEvent(ev);
                updateNeurons(ev.timestamp());
            } else {
                transmitEvent(event);
                updateNeurons(event.timestamp());
            }

            if (static_cast<double>(event.timestamp()) - m_saveTime.display > static_cast<size_t>(5 * E6)) {
                m_saveTime.display = static_cast<double>(event.timestamp());
                std::cout << static_cast<int>(static_cast<double>(100 * event.timestamp()) / m_endTime) << "%" << std::endl;
    //            m_spinet.intermediateSave(m_saveCount);
    //            ++m_saveCount;
            }
    }
}

/**
 * @param time
 */
void NetworkHandle::updateNeurons(size_t time) {
    if (static_cast<double>(time) - m_saveTime.update > m_networkConf.getMeasurementInterval()) {
        m_totalNbEvents += m_countEvents;
        m_spinet.updateNeuronsStates(static_cast<long>(static_cast<double>(time) - m_saveTime.update));
        auto alpha = 0.6;
        m_averageEventRate = (alpha * static_cast<double>(m_countEvents)) + (1.0 - alpha) * m_averageEventRate;
        m_countEvents = 0;
        m_saveData["eventRate"].push_back(m_averageEventRate);
        m_saveData["networkRate"].push_back(m_spinet.getAverageActivity());
        m_saveTime.update = static_cast<double>(time);
    }
}

/**
 *
 */
void NetworkHandle::resetAllNeurons() {
    m_spinet.resetAllNeurons();
    m_saveTime.update = 0;
    m_saveData["eventRate"].clear();
    m_saveData["networkRate"].clear();
}

/**
 *
 */
void NetworkHandle::load() {
    std::string fileName;
    fileName = getNetworkConfig().getNetworkPath() + "networkState.json";

    nlohmann::json state;
    std::ifstream ifs(fileName);
    if (ifs.is_open()) {
        try {
            ifs >> state;
            m_saveCount = state["save_count"];
            m_totalNbEvents = state["nb_events"];
        } catch (const std::exception &e) {
            std::cerr << "In network state file: " << fileName << e.what() << std::endl;
        }
    }
    m_spinet.loadWeights();
}

void NetworkHandle::intermediateSave(size_t nbRun) {
    std::cout << "Saving Intermediate Network..." << std::endl;
    m_spinet.intermediateSave(nbRun);
    std::cout << "Finished." << std::endl;
}

/**
 * Saves information about the network actual state.
 * @param eventFileName - Name of the event file.
 * @param nbRun - Number of time the event file has been shown.
 */
void NetworkHandle::save(const std::string &eventFileName = "", const size_t nbRun = 0) {
    std::cout << "Saving Network..." << std::endl;
    std::string fileName;
    fileName = m_networkConf.getNetworkPath() + "networkState.json";
    m_iteration = 0;

    nlohmann::json state;
    std::ifstream ifs(fileName);
    resetAllNeurons();
    if (ifs.is_open()) {
        try {
            ifs >> state;
            auto eventFileNames = static_cast<std::vector<std::string>>(state["event_file_name"]);
            auto nbRuns = static_cast<std::vector<std::size_t>>(state["nb_run"]);
            eventFileNames.push_back(eventFileName);
            nbRuns.push_back(nbRun);
            state["event_file_name"] = eventFileNames;
            state["nb_run"] = nbRuns;
        } catch (const std::exception &e) {
            std::cerr << "In network state file: " << fileName << e.what() << std::endl;
        }
    } else {
        std::cout << "Creating network state file" << std::endl;
        std::vector<std::string> fileNames = {eventFileName};
        std::vector<size_t> nbRuns = {nbRun};
        state["event_file_name"] = fileNames;
        state["nb_run"] = nbRuns;
    }

    state["learning_data"] = m_saveData;
    state["save_count"] = m_saveCount;
    state["nb_events"] = m_totalNbEvents;

    std::ofstream ofs(fileName);
    if (ofs.is_open()) {
        ofs << std::setw(4) << state << std::endl;
    } else {
        std::cout << "cannot save network state file" << std::endl;
    }
    ofs.close();

    m_spinet.saveNetwork();

    std::cout << "Finished." << std::endl;
}

/**
 *
 * @param sequence
 */
void NetworkHandle::saveStatistics(size_t simulation, size_t sequence, const std::string& folderName, bool reset, bool sep_speed, int n_speed) {
    std::cout << "Starting saving the statistics..." << std::endl;
    if(sequence==-1) {
        m_spinet.saveOrientations();
    }
    else {
        m_iteration = 0;
        if(m_spinet.getEventsParameters()==0) {
            std::cout << "yo?" << std::endl;
            m_spinet.saveStatistics(simulation, sequence, folderName, sep_speed, n_speed);
        }
    }
    resetAllNeurons();
    if(reset) {
        m_spinet.resetSTrain();
    }
    std::cout << "Finished." << std::endl;
}

/**
 *
 * @param event
 */
void NetworkHandle::transmitEvent(const Event &event) {
    // if(event.x() >= 150 && event.x() < 216 && event.y() >= 100 && event.y() < 166) {
        ++m_countEvents;
    // }
    //if(event.timestamp() <= 500000) {
        m_spinet.addEvent(event);
    //}
}

/**
 *
 * @param time
 * @param id
 * @param layer
 */
void NetworkHandle::trackNeuron(const long time, const size_t id, const size_t layer) {
    if (m_simpleNeuronConf.TRACKING == "partial") {
        if (m_spinet.getNetworkStructure()[layer] > 0) {
            m_spinet.getNeuron(id, layer).get().trackPotential(time);
        }
    }
}

void NetworkHandle::changeNeuronToTrack(int n_x, int n_y) {
    m_spinet.changeTrack(n_x, n_y);
}

/**
 *
 * @param idNeuron
 * @param layer
 * @param camera
 * @param synapse
 * @param z
 * @return
 */
cv::Mat NetworkHandle::neuronWeightMatrix(size_t idNeuron, size_t layer, size_t camera, size_t synapse, size_t z) {
    if (m_spinet.getNetworkStructure()[layer] > 0) {
        auto dim = getNetworkConfig().getLayerConnectivity()[layer].neuronSizes[0];
        auto dim_prev = dim;
        auto dim_prev2 = dim;
        if(layer > 0) {
            dim_prev = getNetworkConfig().getLayerConnectivity()[layer-1].neuronSizes[0];
            if(layer > 1) {
                dim_prev2 = getNetworkConfig().getLayerConnectivity()[layer-2].neuronSizes[0];
            }
        }
        cv::Mat weightImage;
        cv::Mat weightImagePrev;
        if(layer < 1) {
            weightImage = cv::Mat::zeros(static_cast<int>(dim[1]), static_cast<int>(dim[0]), CV_8UC3);
        }
        else if(layer ==1) {
            weightImage = cv::Mat::zeros(static_cast<int>(dim[1] * dim_prev[1] ), static_cast<int>(dim[0] * dim_prev[0]), CV_8UC3);
        }
        else {
            weightImage = cv::Mat::zeros(static_cast<int>(dim[1] * dim_prev[1] * dim_prev2[1]), static_cast<int>(dim[0] * dim_prev[0] * dim_prev2[1]), CV_8UC3);

        }
        double weight;
        for (size_t x = 0; x < dim[0]; ++x) {
            for (size_t y = 0; y < dim[1]; ++y) {
                if (layer == 0) {
                    for (size_t p = 0; p < NBPOLARITY; p++) {
                        weight = m_spinet.getNeuron(idNeuron, layer).get().getWeightsMatrix().get(p, camera, synapse, x, y) * 255;
                        weightImage.at<cv::Vec3b>(static_cast<int>(y), static_cast<int>(x))[static_cast<int>(2 - p)] = static_cast<unsigned char>(weight);
                    }
                } else if(layer == 1) {
                    auto weightDim = m_spinet.getNeuron(idNeuron, layer).get().getWeightsDimension();
                    auto weightDimPrev_x = m_networkConf.getLayerConnectivity()[layer-1].sizes[0];
                    auto weightDimPrev_y = m_networkConf.getLayerConnectivity()[layer-1].sizes[1];
                    auto weightDimPrev_z = m_networkConf.getLayerConnectivity()[layer-1].sizes[2];

                    double maximum_wi = m_spinet.getNeuron(idNeuron, layer).get().getWeightsMatrix().getMaximum();
                    double minimum_wi = m_spinet.getNeuron(idNeuron, layer).get().getWeightsMatrix().getMinimum();

                    for(size_t dimx = 0; dimx < weightDim[0]; ++dimx) {
                        for(size_t dimy = 0; dimy < weightDim[1]; ++dimy) {
                            int indNeurMax = m_spinet.getNeuron(idNeuron, layer).get().getWeightsMatrix().getNMaximum(dimx, dimy, z);
                            int indPreviousLayerNeur = (dimy + m_spinet.getNeuron(idNeuron, layer).get().getOffset().y()) * weightDimPrev_z
                                                    + (dimx + m_spinet.getNeuron(idNeuron, layer).get().getOffset().x()) * weightDimPrev_y * weightDimPrev_z
                                                    + indNeurMax;
                            auto weightImagePrev = neuronWeightMatrix(indPreviousLayerNeur, layer-1, 0, 0, 0);
                            for(int yv = 0; yv < dim_prev[1]; ++yv) {
                                for(int xv = 0; xv < dim_prev[0]; ++xv) {
                                    for (size_t p = 0; p < 2; p++) {
                                        weightImage.at<cv::Vec3b>(static_cast<int>(yv + dimy * dim_prev[1]), 
                                    static_cast<int>(xv + dimx * dim_prev[0]))[static_cast<int>(2 - p)] = weightImagePrev.at<cv::Vec3b>(static_cast<int>(yv), static_cast<int>(xv))[static_cast<int>(2 - p)] * (m_spinet.getNeuron(idNeuron, layer).get().getWeightsMatrix().get(dimx, dimy, indNeurMax) - minimum_wi ) / (maximum_wi - minimum_wi);
                                    }   
                                }
                            }
                        }
                    }
                    
                }
                else {
                    // std::cout << "layer = " << layer << std::endl;
                    auto weightDim = m_spinet.getNeuron(idNeuron, layer).get().getWeightsDimension();
                    auto weightDimPrev_x = m_networkConf.getLayerConnectivity()[layer-1].sizes[0];
                    auto weightDimPrev_y = m_networkConf.getLayerConnectivity()[layer-1].sizes[1];
                    auto weightDimPrev_z = m_networkConf.getLayerConnectivity()[layer-1].sizes[2];
                    double maximum_wi = m_spinet.getNeuron(idNeuron, layer).get().getWeightsMatrix().getMaximum();
                    double minimum_wi = m_spinet.getNeuron(idNeuron, layer).get().getWeightsMatrix().getMinimum();
                    for(size_t dimx = 0; dimx < weightDim[0]; ++dimx) {
                        for(size_t dimy = 0; dimy < weightDim[1]; ++dimy) {
                            int indNeurMax = m_spinet.getNeuron(idNeuron, layer).get().getWeightsMatrix().getNMaximum(dimx, dimy, z);
                            int indPreviousLayerNeur = (dimy + m_spinet.getNeuron(idNeuron, layer).get().getOffset().y()) * weightDimPrev_z
                                                    + (dimx + m_spinet.getNeuron(idNeuron, layer).get().getOffset().x()) * weightDimPrev_y * weightDimPrev_z
                                                    + indNeurMax;
                            auto weightImagePrev = neuronWeightMatrix(indPreviousLayerNeur, layer-1, 0, 0, 0);
                            for(int yv = 0; yv < dim_prev[1]; ++yv) {
                                for(int xv = 0; xv < dim_prev[0]; ++xv) {
                                    for(int yv2 = 0; yv2 < dim_prev2[1]; ++yv2) {
                                        for(int xv2 = 0; xv2 < dim_prev2[0]; ++xv2) {
                                            for (size_t p = 0; p < 2; p++) {
                                                weightImage.at<cv::Vec3b>(static_cast<int>(yv2 + yv * dim_prev2[1] + dimy * dim_prev[1] * dim_prev2[1]), 
                                            static_cast<int>(xv2 + xv * dim_prev2[0] + dimx * dim_prev[0] * dim_prev2[0]))[static_cast<int>(2 - p)] = weightImagePrev.at<cv::Vec3b>(static_cast<int>(yv2 + yv * dim_prev2[1]), 
                                    static_cast<int>(xv2 + xv * dim_prev2[0]))[static_cast<int>(2 - p)] * (m_spinet.getNeuron(idNeuron, layer).get().getWeightsMatrix().get(dimx, dimy, indNeurMax) - minimum_wi ) / (maximum_wi - minimum_wi);
                                            }   
                                        }
                                    }
                                }
                            }
                        }
                    }

                }
            }
        }
        double min, max;
        minMaxIdx(weightImage, &min, &max);
        cv::Mat weights = weightImage;
        if(layer == 0) {
            cv::Mat weights = weightImage * 255 / max;
        }
        return weights;
    }
    return cv::Mat::zeros(0, 0, CV_8UC3);
}

/**
 *
 * @param idNeuron
 * @param layer
 * @return
 */
cv::Mat NetworkHandle::getSummedWeightNeuron(size_t idNeuron, size_t layer) {
    if (m_spinet.getNetworkStructure()[layer] > 0) {
        return m_spinet.getNeuron(idNeuron, layer).get().summedWeightMatrix();
    }
    return cv::Mat::zeros(0, 0, CV_8UC3);
}

/**
 * Opens an event file (in the npz format) and load all the events in memory into a vector.
 * If nbPass is greater than 1, the events are concatenated multiple times and the timestamps are updated in accordance.
 * Returns the vector of events.
 * Only for loadNpzEvents camera event files.
 * @param events
 * @param nbPass
 */
void NetworkHandle::loadNpzEvents(std::vector<Event> &events, size_t nbPass) {
    events.clear();
    size_t pass, count;

    cnpy::NpyArray timestamps_array = cnpy::npz_load(m_eventsPath, "arr_0");
    cnpy::NpyArray x_array = cnpy::npz_load(m_eventsPath, "arr_1");
    cnpy::NpyArray y_array = cnpy::npz_load(m_eventsPath, "arr_2");
    cnpy::NpyArray polarities_array = cnpy::npz_load(m_eventsPath, "arr_3");
    
    size_t sizeArray = timestamps_array.shape[0];



    auto *ptr_timestamps = timestamps_array.data<long>();
    std::vector<long> timestamps(ptr_timestamps, ptr_timestamps + sizeArray);
    auto *ptr_x = x_array.data<int16_t>();
    std::vector<int16_t> x(ptr_x, ptr_x + sizeArray);
    auto *ptr_y = y_array.data<int16_t>();
    std::vector<int16_t> y(ptr_y, ptr_y + sizeArray);
    auto *ptr_polarities = polarities_array.data<bool>();
    std::vector<bool> polarities(ptr_polarities, ptr_polarities + sizeArray);

    auto cameras = std::vector<bool>(sizeArray, false);
    try {
        cnpy::NpyArray cameras_array = cnpy::npz_load(m_eventsPath, "arr_4");
        auto *ptr_cameras = cameras_array.data<bool>();
        cameras = std::vector<bool>(ptr_cameras, ptr_cameras + sizeArray);
    } catch (const std::exception &e) {
        std::cout << "NPZ file: " << e.what() << " defaulting to 1 camera.\n";
    }

    long firstTimestamp = timestamps[0];
    long lastTimestamp = static_cast<long>(timestamps[sizeArray - 1]);
    std::cout << "Initial duration = " << lastTimestamp << std::endl;
    Event event{};
    for (pass = 0; pass < static_cast<size_t>(nbPass); ++pass) {
        for (count = 0; count < sizeArray; ++count) {
            event = Event(timestamps[count] + static_cast<long>(pass) * (lastTimestamp - firstTimestamp),
                          x[count],
                          y[count],
                          polarities[count],
                          cameras[count]);
            events.push_back(event);
        }
    }
    m_nbEvents = nbPass * sizeArray;
}

/**
 *
 * @param events
 * @return
 */
bool NetworkHandle::loadHDF5Events(std::vector<Event> &events, size_t nbPass) {
    if (m_eventFile.offset >= m_eventFile.dims) {
        m_eventFile.packetSize = 10000;
        ++m_eventFile.countPass;
        m_eventFile.offset = 0;
        if (m_eventFile.countPass >= nbPass) {
            return true;
        }
    } else if (m_eventFile.offset + m_eventFile.packetSize > m_eventFile.dims) {
        m_eventFile.packetSize = m_eventFile.dims - m_eventFile.offset;
    }

    events.clear();
    auto vT = std::vector<uint64_t>(m_eventFile.packetSize);
    auto vX = std::vector<uint16_t>(m_eventFile.packetSize);
    auto vY = std::vector<uint16_t>(m_eventFile.packetSize);
    auto vP = std::vector<uint8_t>(m_eventFile.packetSize);
    auto vC = std::vector<uint8_t>(m_eventFile.packetSize);

    H5::DataSpace filespace = m_eventFile.timestamps.getSpace();
    hsize_t dim[1] = {m_eventFile.packetSize};
    H5::DataSpace memspace(1, dim);
    filespace.selectHyperslab(H5S_SELECT_SET, &m_eventFile.packetSize, &m_eventFile.offset);

    m_eventFile.timestamps.read(vT.data(), H5::PredType::NATIVE_UINT64, memspace, filespace);
    m_eventFile.x.read(vX.data(), H5::PredType::NATIVE_UINT16, memspace, filespace);
    m_eventFile.y.read(vY.data(), H5::PredType::NATIVE_UINT16, memspace, filespace);
    m_eventFile.polarities.read(vP.data(), H5::PredType::NATIVE_UINT8, memspace, filespace);
    m_eventFile.cameras.read(vC.data(), H5::PredType::NATIVE_UINT8, memspace, filespace);

    auto vTOffset = m_eventFile.countPass * (m_eventFile.lastTimestamp - m_eventFile.firstTimestamp);
    for (int i = 0; i < m_eventFile.packetSize; i++) {
        if (getNetworkConfig().getNbCameras() == 2 || vC[i] == 0) {
            events.emplace_back(vT[i] - m_eventFile.firstTimestamp + vTOffset, vX[i], vY[i], vP[i], vC[i]);
        }
    }

    m_eventFile.offset += m_eventFile.packetSize;
    return false;
}

/**
 *
 */
void NetworkHandle::readFirstAndLastTimestamp() {
    H5::DataSpace filespace = m_eventFile.timestamps.getSpace();
    hsize_t dim[1] = {1};
    H5::DataSpace memspace(1, dim);
    hsize_t count = 1;

    hsize_t offset = 0;
    auto first = std::vector<uint64_t>(1);
    filespace.selectHyperslab(H5S_SELECT_SET, &count, &offset);
    m_eventFile.timestamps.read(first.data(), H5::PredType::NATIVE_UINT64, memspace, filespace);

    offset = m_eventFile.dims - 1;
    auto last = std::vector<uint64_t>(1);
    filespace.selectHyperslab(H5S_SELECT_SET, &count, &offset);
    m_eventFile.timestamps.read(last.data(), H5::PredType::NATIVE_UINT64, memspace, filespace);

    m_eventFile.firstTimestamp = first[0];
    m_eventFile.lastTimestamp = last[0];
}

void NetworkHandle::normalizeL1Weights() {
    m_spinet.normalizeWeights();
}

void NetworkHandle::lateralRandom(int norm_factor) {
    m_spinet.randomLateralInhibition(norm_factor);
}

void NetworkHandle::inhibitionShuffle(int case_) {
    m_spinet.shuffleInhibition(case_);
}

void NetworkHandle::assignOrientation(int z, int ori, int thickness) {
    m_spinet.assignOrientations(z, ori, thickness);
}

void NetworkHandle::assignComplexOrientation(int neur, int ori, int thickness) {
    m_spinet.assignComplexOrientations(neur, ori, thickness);
}

void NetworkHandle::assignPatchSize(double patch) {
    m_spinet.assignPatchSize(patch);
}

void NetworkHandle::setSequenceParameters(std::vector<std::vector<size_t>> parameters) {
    m_spinet.setEventsParameters(parameters);
}

void NetworkHandle::deactivateDynamicInhib(bool activation) {
    m_spinet.setDynamicActivation(activation);
}