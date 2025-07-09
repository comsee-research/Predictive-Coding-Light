//
// Created by Thomas on 14/04/2021.
//

#include "Neuron.hpp"

/**
 * Defines an abstract neuron.
 * @param index - unique id.
 * @param layer - depth of the neuron in the spiking network.
 * @param conf - configuration file.
 * @param pos - indicates the position of the neuron relative to the other neurons of the layer.
 * @param offset - removed to the position in order to access the correct weights of the neuron.
 */
Neuron::Neuron(size_t index, size_t layer, NeuronConfig conf, Position pos, Position offset) :
        m_index(index),
        m_layer(layer),
        m_conf(conf),
        m_pos(pos),
        m_offset(offset),
        m_spikeCounter(0),
        m_lateralLocalInhibitionEvents(boost::circular_buffer<NeuronEvent>(1000)),
        m_range(std::vector<size_t>(2, 1)),
        m_trackingSpikeTrain(std::vector<size_t>(0)),
        m_amount_of_events(std::vector<size_t>(6, 0)),
        m_negativeLimits(-20) {
    m_threshold = conf.VTHRESH;
    m_decay = 1.0;
    m_spike = false;
    m_adaptationPotential = 0;
    m_eLTP = m_conf.ETA_LTP * m_conf.NORM_FACTOR;
    m_eLTD = m_conf.ETA_LTD * m_conf.NORM_FACTOR;
    m_iLTP = m_conf.ETA_ILTP * m_conf.ETA_INH;
    m_iLTD = m_conf.ETA_ILTD * m_conf.ETA_INH;
}

/**
 * Returns the neuron's potential after the decay depending on the time passed from the last event.
 * @param time
 * @return
 */
inline double Neuron::getPotential(size_t time) {
    return m_potential * exp(-static_cast<double>(time - m_timestampLastEvent) / m_conf.TAU_M);
}

/**
 *
 * @param time
 * @return
 */
inline double Neuron::refractoryPotential(size_t time) {
    if(m_spikingTime==0) {
        return 0;
    }    
    else {
        return m_conf.DELTA_RP * exp(-static_cast<double>(time - m_spikingTime) / m_conf.TAU_RP);
    }
}

/**
 *
 * @param time
 */
inline void Neuron::potentialDecay(size_t time) {
    m_potential *= exp(-static_cast<double>(time - m_timestampLastEvent) / m_conf.TAU_M);
}

/**
 *
 * @param time
 */
inline void Neuron::adaptationPotentialDecay(size_t time) {
    m_adaptationPotential *= exp(-static_cast<double>(time - m_timestampLastEvent) / m_conf.TAU_SRA);
    // std::cout << "m_adaptation Potential = " << m_adaptationPotential << std::endl;
}

/**
 * Keeps track of the potential of the neuron.
 * @param potential
 */
void Neuron::assignToPotentialTrain(std::pair<double, uint64_t> potential) {
    m_trackingPotentialTrain.push_back(potential);
}

/**
 * Assigns the actual threshold of the neuron to the list of thresholds. 
 * This function is useful if the threshold isn't static.
 */
void Neuron::assignToPotentialThreshold() {
    m_potentialThreshold.push_back(m_threshold);
}

/**
 * Updates the total number of events received, depending on its type (excitatory or inhibitory).
 * @param type
 */
void Neuron::assignToAmountOfEvents(int type) {
    if (m_amount_of_events.size() == 0) {
        for (int i = 0; i < 6; i++) {
            m_amount_of_events.push_back(0);
        }
    }
    m_amount_of_events[type] += 1;
}

/**
 * Updates the amount of lateral inhibition received from neurons at the same position in the actual layer.
 * @param type
 * @param pos
 * @param wi
 */
void Neuron::assignToSumLateralWeights(int type, Position pos, double wi, size_t depth) {
    int m_range_x = m_range[0];
    int m_range_y = m_range[1];
    if (m_sumOfInhibWeights.size() == 0) {
        for (int i = 0; i < 2; i++) {
            std::vector<std::vector<double>> temp;
            m_sumOfInhibWeights.push_back(temp);
            for(int d = 0; d < depth; d++) {
                std::vector<double> final_temp;
                m_sumOfInhibWeights[i].push_back(final_temp);
                for (int j = static_cast<int>(-m_range_x); j < static_cast<int>(m_range_x + 1); j++) {
                    for (int k = static_cast<int>(-m_range_y); k < static_cast<int>(m_range_y + 1); k++) {
                        if (j == 0 && k == 0) {
                            continue;
                        } else {
                            m_sumOfInhibWeights[i][d].push_back(0);
                        }
                    }
                }
            }
        }
    }
    int x_border_min = m_pos.x() - m_range_x;
    int y_border_min = m_pos.y() - m_range_y;
    int position;
    if ((pos.x() > m_pos.x()) || (pos.x() == m_pos.x() && pos.y() > m_pos.y())) {
        position = (2 * (m_range_y) + 1) * (pos.x() - x_border_min) + (pos.y() - y_border_min) - 1;
    } else {
        position = (2 * (m_range_y) + 1) * (pos.x() - x_border_min) + (pos.y() - y_border_min);
    }
    m_sumOfInhibWeights[type][pos.z()][position] += wi;
}

/**
 * Updates the amount of lateral inhibition received from neurons at the same position in the actual layer.
 * @param type
 * @param pos
 * @param wi
 */
void Neuron::assignToSumLateralExcitatoryWeights(int type, Position pos, double wi, size_t depth) {
    int m_range_x = m_range[0];
    int m_range_y = m_range[1];
    if (m_sumOfExcitWeights.size() == 0) {
        for (int i = 0; i < 2; i++) {
            std::vector<std::vector<double>> temp;
            m_sumOfExcitWeights.push_back(temp);
            for(int d = 0; d < depth; d++) {
                std::vector<double> final_temp;
                m_sumOfExcitWeights[i].push_back(final_temp);
                for (int j = static_cast<int>(-m_range_x); j < static_cast<int>(m_range_x + 1); j++) {
                    for (int k = static_cast<int>(-m_range_y); k < static_cast<int>(m_range_y + 1); k++) {
                        if (j == 0 && k == 0) {
                            continue;
                        } else {
                            m_sumOfExcitWeights[i][d].push_back(0);
                        }
                    }
                }
            }
        }
    }
    int x_border_min = m_pos.x() - m_range_x;
    int y_border_min = m_pos.y() - m_range_y;
    int position;
    if ((pos.x() > m_pos.x()) || (pos.x() == m_pos.x() && pos.y() > m_pos.y())) {
        position = (2 * (m_range_y) + 1) * (pos.x() - x_border_min) + (pos.y() - y_border_min) - 1;
    } else {
        position = (2 * (m_range_y) + 1) * (pos.x() - x_border_min) + (pos.y() - y_border_min);
    }
    type = 1;
    m_sumOfExcitWeights[type][pos.z()][position] += wi;
}

/**
 * Updates the amount of top-down inhibition received from neurons at the same position in the next layer.
 * @param index 
 * @param wi 
 * @param depth 
 */
void Neuron::assignToSumTopDownWeights(int index, double wi, size_t depth) {
    int size = m_outConnections.size() / depth;
    if (m_sumOfTopDownWeights.size() == 0) {
        m_start.clear();
        for (int i = 0; i < size; i++) {
            std::vector<double> temp;
            m_start.push_back(m_outConnections[i*depth].get().getIndex());
                for (int j = 0; j < depth; j++) {
                    temp.push_back(0);
                }
            m_sumOfTopDownWeights.push_back(temp);
        }
    }
    int actual_start = 0;
    for(int i = 0; i<size; i++) {
        if(index >= m_start[i] && index < m_start[i]+depth) {
            actual_start = i;
            break;
        }
    }
    m_sumOfTopDownWeights[actual_start][index-m_start[actual_start]] += wi;
}

/**
 * Updates the amount of top-down inhibition received from neurons at the same position in the next layer.
 * @param index 
 * @param wi 
 * @param depth 
 */
void Neuron::assignToSumTopDownExcitatoryWeights(int index, double wi, size_t depth) {
    int size = m_outConnections.size() / depth;
    if (m_sumOfTopDownExcitWeights.size() == 0) {
        m_start.clear();
        for (int i = 0; i < size; i++) {
            std::vector<double> temp;
            m_start.push_back(m_outConnections[i*depth].get().getIndex());
                for (int j = 0; j < depth; j++) {
                    temp.push_back(0);
                }
            m_sumOfTopDownExcitWeights.push_back(temp);
        }
    }
    int actual_start = 0;
    for(int i = 0; i<size; i++) {
        if(index >= m_start[i] && index < m_start[i]+depth) {
            actual_start = i;
            break;
        }
    }
    m_sumOfTopDownExcitWeights[actual_start][index-m_start[actual_start]] += wi;
}

/**
 * Keeps track of the modification of the potential by inhibitory events.
 * @param type
 * @param variation
 */
void Neuron::assignToTimingOfInhibition(int type, std::tuple<double, double, uint64_t> variation) {
    if (m_timingOfInhibition.size() == 0) {
        std::vector<std::tuple<double, double, uint64_t>> temp;
        for (int i = 0; i < 3; i++) {
            m_timingOfInhibition.push_back(temp);
        }
    }
    m_timingOfInhibition[type].push_back(variation);
}

/**
 *
 * @return
 * Keeps track of the modification of the potential by excitatory events. 
 * @param event 
 */
void Neuron::assignToExcitatoryEvents(std::tuple<double, uint64_t> event) {
    m_excitatoryEvents.push_back(event);
}

/**
 * Sets the range of the lateral inhibition of the neuron.
 * @param inhibitionRange 
 */
void Neuron::setInhibitionRange(std::vector<size_t> inhibitionRange) {
    m_range = inhibitionRange;
}

/**
 * Returns the train of potentials.
 * @return std::vector<std::pair<double, uint64_t>> 
 */
std::vector<std::pair<double, uint64_t>> Neuron::getPotentialTrain() {        
    return m_trackingPotentialTrain;
}

/**
 *
 * @return
 * Returns the train of thresholds.
 * @return std::vector<double>
 */
std::vector<double> Neuron::getPotentialThreshold() {
    return m_potentialThreshold;
}

/**
 *
 * @return
 * Returns the vector containing the number of events per type.
 * @return std::vector<size_t>
 */
std::vector<size_t> Neuron::getAmountOfEvents() {
    return m_amount_of_events;
}

/**
 * Returns the amount of lateral inhibition received by the neuron from different neurons at the same layer.
 * @return std::vector<std::vector<double>> 
 */
std::vector<std::vector<std::vector<double>>> Neuron::getSumLateralWeights() {
    return m_sumOfInhibWeights;
}

/**
 * Returns the amount of top-down inhibition received by the neuron from different neurons from the next layer.
 * @return std::vector<std::vector<double>> 
 */
std::vector<std::vector<double>> Neuron::getSumTopDownWeights() {
    return m_sumOfTopDownWeights;
}

/**
 * Returns the amount of lateral inhibition received by the neuron from different neurons at the same layer.
 * @return std::vector<std::vector<double>> 
 */
std::vector<std::vector<std::vector<double>>> Neuron::getSumLateralExcitatoryWeights() {
    return m_sumOfExcitWeights;
}

/**
 * Returns the amount of top-down inhibition received by the neuron from different neurons from the next layer.
 * @return std::vector<std::vector<double>> 
 */
std::vector<std::vector<double>> Neuron::getSumTopDownExcitatoryWeights() {
    return m_sumOfTopDownExcitWeights;
}


/**
 * Returns the potentials modified from inhibitory events.
 * @return std::vector<std::vector<std::tuple<double, double, uint64_t>>>
 */
std::vector<std::vector<std::tuple<double, double, uint64_t>>> Neuron::getTimingOfInhibition() {
    return m_timingOfInhibition;
}

/**
 * Returns the potentials modified from the excitatory events.
 * @return std::vector<std::tuple<double, uint64_t>>
 */
std::vector<std::tuple<double, uint64_t>> Neuron::getExcitatoryEvents() {
    return m_excitatoryEvents;
}

/**
* Computes the neuron's lifespan as well as the average spiking rate.
* @param timeInterval
*/
void Neuron::updateState(size_t timeInterval) {
    m_lifeSpan += timeInterval;
    double spikesPerSecond = static_cast<double>(m_spikeRateCounter) * (E6 / static_cast<double>(timeInterval)); // spikes/s
    auto alpha = 0.6;
    m_spikingRateAverage = (alpha * spikesPerSecond) + (1.0 - alpha) * m_spikingRateAverage; // exponential rolling average
    m_spikeRateCounter = 0;
    int i = 0;
}

/**
 *
 * @param count
 */
void Neuron::learningDecay(double count) {
    m_decay = 1 / (1 + m_conf.DECAY_RATE * count);
}

/**
 * Rescales the neuron's threshold depending on the deviation from the average spike rate and a target spike rate.
 */
inline void Neuron::thresholdAdaptation() {
    if (m_spikingRateAverage > m_conf.TARGET_SPIKE_RATE) {
        m_threshold += m_conf.ETA_SR * (1 - exp(m_conf.TARGET_SPIKE_RATE - m_spikingRateAverage));
    } else {
        m_threshold -= m_conf.ETA_SR * (1 - exp(m_spikingRateAverage - m_conf.TARGET_SPIKE_RATE));
    }

    if (m_threshold < m_conf.MIN_THRESH) {
        m_threshold = m_conf.MIN_THRESH;
    }
}

/**
 *
 */
inline void Neuron::spikeRateAdaptation() {
    m_adaptationPotential += m_conf.DELTA_SRA;
}

void Neuron::checkNegativeLimits() {
    if(m_potential < m_negativeLimits) {
        m_potential = m_negativeLimits;
    }
}

/**
 * Sets the value of the potential before it gets inhibited.
 */
void Neuron::setLastBeforeInhibitionPotential() {
    m_beforeInhibitionPotential = m_potential;
}

/**
 *
 * @param filePath
 */
void Neuron::saveState(std::string &filePath) {
    nlohmann::json state;

    writeJson(state);

    std::ofstream ofs(filePath + std::to_string(m_index) + ".json");
    if (ofs.is_open()) {
        ofs << std::setw(4) << state << std::endl;
    } else {
        std::cout << "cannot save neuron state file" << std::endl;
    }
    ofs.close();
}

/**
 *
 * @param filePath
 */
void Neuron::loadState(std::string &filePath) {
    nlohmann::json state;
    std::ifstream ifs(filePath + std::to_string(m_index) + ".json");
    if (ifs.is_open()) {
        try {
            ifs >> state;
        } catch (const std::exception &e) {
            std::cerr << "In Neuron state file: " << filePath + ".json" << std::endl;
            throw;
        }
        readJson(state);
    } else {
        std::cout << "cannot open neuron state file" << std::endl;
    }
    ifs.close();
}

/**
 *
 * @param state
 */
void Neuron::writeJson(nlohmann::json &state) {
    std::vector<size_t> position = {m_pos.x(), m_pos.y(), m_pos.z()};
    std::vector<size_t> offset = {m_offset.x(), m_offset.y(), m_offset.z()};
    std::vector<size_t> inIndex;
    std::vector<size_t> outIndex;
    std::vector<size_t> staticInhibitionIndex;
    std::vector<size_t> tdInhibitionIndex;
    std::vector<size_t> liInhibitionIndex;
    for (auto neuron: m_inConnections) {
        inIndex.push_back(neuron.get().getIndex());
    }
    for (auto neuron: m_outConnections) {
        outIndex.push_back(neuron.get().getIndex());
    }
    for (auto neuron: m_lateralLocalInhibitionConnections) {
        staticInhibitionIndex.push_back(neuron.get().getIndex());
    }
    for (auto neuron: m_topDownDynamicInhibitionConnections) {
        tdInhibitionIndex.push_back(neuron.get().getIndex());
    }
    for (auto neuron: m_lateralDynamicInhibitionConnections) {
        liInhibitionIndex.push_back(neuron.get().getIndex());
    }
    state["position"] = position;
    state["offset"] = offset;
    state["in_connections"] = inIndex;
    state["out_connections"] = outIndex;
    state["static_inhibition"] = staticInhibitionIndex;
    state["topdown_dynamic_inhibition"] = tdInhibitionIndex;
    state["lateral_dynamic_inhibition"] = liInhibitionIndex;
    state["potential"] = m_potential;
    state["count_spike"] = m_totalSpike;
    state["threshold"] = m_threshold;
    state["lifespan"] = m_lifeSpan;
    state["spiking_rate"] = m_spikingRateAverage;
    state["learning_decay"] = m_decay;
    state["spike_train"] = m_trackingSpikeTrain;
}

/**
 *
 * @param state
 */
void Neuron::readJson(const nlohmann::json &state) {
    m_totalSpike = state["count_spike"];
//    m_threshold = state["threshold"];
//    m_lifeSpan = state["lifespan"];
    m_decay = state["learning_decay"];
    m_spikingRateAverage = state["spiking_rate"];
//    m_potential = state["potential"];
}

/**
 *
 * @param time
 */
void Neuron::trackPotential(const size_t time) {
    double potential = getPotential(time);
    m_trackingPotentialTrain.emplace_back(potential, time);
}

/**
 *
 * @return
 */
size_t Neuron::getActivityCount() {
    return m_activityCounter;
}

/**
 *
 */
void Neuron::resetActivityCount() {
    m_activityCounter = 0;
}

/**
 *
 * @param neuron
 */
void Neuron::addOutConnection(Neuron &neuron) {
    m_outConnections.emplace_back(neuron);
}

/**
 *
 * @param neuron
 */
void Neuron::addInConnection(Neuron &neuron) {
    m_inConnections.emplace_back(neuron);
}

/**
 *
 * @param id
 */
void Neuron::initInWeights(size_t id) {
    m_weights.addWeight(id, true);
}

/**
 * @param neuron
 */
void Neuron::addTopDownDynamicInhibitionConnection(Neuron &neuron) {
    m_topDownDynamicInhibitionConnections.emplace_back(neuron);
}

/**
 *
 * @param neuron
 */
void Neuron::initTopDownDynamicInhibitionWeights(size_t id) {
    m_topDownInhibitionWeights.addWeight(id, true);
}

/**
 *
 * @param neuron
 */
void Neuron::initTopDownDynamicExcitationWeights(size_t id) {
    m_topDownExcitationWeights.addWeight(id, true);
}

/**
 *
 * @param neuron
 */
void Neuron::addLateralLocalInhibitionConnections(Neuron &neuron) {
    m_lateralLocalInhibitionConnections.emplace_back(neuron);
}

/**
 *
 * @param neuron
 */
void Neuron::addLateralDynamicInhibitionConnections(Neuron &neuron) {
    m_lateralDynamicInhibitionConnections.emplace_back(neuron);
}

/**
 *
 * @param id
 */
void Neuron::initLateralDynamicInhibitionWeights(size_t id) {
    m_lateralInhibitionWeights.addWeight(id, true);
}

/**
 *
 * @param id
 */
void Neuron::initLateralDynamicExcitationWeights(size_t id) {
    m_lateralExcitationWeights.addWeight(id, true);
}

/**
 * Reset to initialization values the different process parameters and statistics of the neuron.
 * This function DOES NOT reset the weights.
 */
void Neuron::resetNeuron() {
    m_potential = 0;
    m_timestampLastEvent = 0;
    m_lastSpikingTime = 0;
    m_spikingTime = 0;
    m_totalSpike = 0;
    m_spikeRateCounter = 0;
    m_activityCounter = 0;
    m_decay = 1.0;
    m_adaptationPotential = 0;
    m_threshold = m_conf.VTHRESH;
    m_spike = false;
    m_firstInput = 0;
    m_lifeSpan = 0;
    m_spikingRateAverage = 0;
    m_trackingSpikeTrain.clear();
    m_trackingPotentialTrain.clear();
    m_potentialThreshold.clear();
    m_amount_of_events.clear();
    for (int i = 0; i < m_sumOfInhibWeights.size(); i++) {
        for(int j = 0; j < m_sumOfInhibWeights[i].size(); j++) {
            m_sumOfInhibWeights[i][j].clear();
        }
    }
    m_sumOfInhibWeights.clear();
    for (int k=0; k<m_sumOfTopDownWeights.size(); k++) {
        m_sumOfTopDownWeights[k].clear();
    }
    m_sumOfTopDownWeights.clear();
    for (int i = 0; i < m_sumOfExcitWeights.size(); i++) {
        for(int j = 0; j < m_sumOfExcitWeights[i].size(); j++) {
            m_sumOfExcitWeights[i][j].clear();
        }
    }
    m_sumOfExcitWeights.clear();
    for (int k=0; k<m_sumOfTopDownExcitWeights.size(); k++) {
        m_sumOfTopDownExcitWeights[k].clear();
    }
    m_sumOfTopDownExcitWeights.clear();
    for (int j = 0; j < m_timingOfInhibition.size(); j++) {
        m_timingOfInhibition[j].clear();
    }    
    m_timingOfInhibition.clear();
    m_excitatoryEvents.clear();
    m_outputs.clear();
}

/**
 * Initializes the lateral inhibition weights randomly from an uniform distribution.
 */
void Neuron::randomInhibition(int norm_factor) {
    std::mt19937 generator(static_cast<unsigned>(std::chrono::system_clock::now().time_since_epoch().count()));
    std::normal_distribution<double> normalDistr(0.0, 1.0);
    std::uniform_real_distribution<double> uniformRealDistr(0.0, 1.0);

    for (auto neuron: m_lateralDynamicInhibitionConnections) {
        m_lateralInhibitionWeights.at(neuron.get().getIndex()) = uniformRealDistr(generator);
    }

    if(norm_factor <= 0) {
        m_lateralInhibitionWeights.normalize(m_conf.LATERAL_NORM_FACTOR);
    }
    else {
        m_lateralInhibitionWeights.normalize(norm_factor);
    }
}

double Neuron::getLateralWeightsAverage() {
    return m_lateralInhibitionWeights.getAverage();
}

double Neuron::getMinimumLateralWeight() {
    return m_lateralInhibitionWeights.getMinimum();
}

double Neuron::getMaximumLateralWeight() {
    return m_lateralInhibitionWeights.getMaximum();
}

/**
 * Shuffles the lateral inhibition weights.
 */
void Neuron::shuffleLateralInhibition() {
    std::vector<double> weights;
    for (auto neuron: m_lateralDynamicInhibitionConnections) {
        weights.push_back(m_lateralInhibitionWeights.at(neuron.get().getIndex()));
    }
    auto rd = std::random_device {};
    auto rng = std::default_random_engine {rd()};
    std::shuffle(std::begin(weights), std::end(weights), rng);
    int i = 0;
    for (auto neuron: m_lateralDynamicInhibitionConnections) {
        m_lateralInhibitionWeights.at(neuron.get().getIndex()) = weights[i];
        i++;
    }
}

/**
 * Shuffles the top-down inhibition weights.
 */
void Neuron::shuffleTopDownInhibition() {
    std::vector<double> weights;
    for (auto neuron: m_topDownDynamicInhibitionConnections) {
        weights.push_back(m_topDownInhibitionWeights.at(neuron.get().getIndex()));
    }
    auto rd = std::random_device {};
    auto rng = std::default_random_engine {rd()};
    std::shuffle(std::begin(weights), std::end(weights), rng);
    int i = 0;
    for (auto neuron: m_topDownDynamicInhibitionConnections) {
        m_topDownInhibitionWeights.at(neuron.get().getIndex()) = weights[i];
        i++;
    }
}

/**
 * Resets the train of spikes.
 */
void Neuron::resetSpikeTrain() {
    m_trackingSpikeTrain.clear();
}
