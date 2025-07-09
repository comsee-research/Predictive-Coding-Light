//
// Created by Thomas on 14/04/2021.
//
#include "SimpleNeuron.hpp"
static std::mt19937 generator_new(time(nullptr));
static std::uniform_int_distribution<int> gen(0,1);

/**
 * Similar to the abstract neuron class.
 * It also takes as input a weight tensor and the number of synapses used when delayed synapses are defined.
 * @param index
 * @param layer
 * @param conf
 * @param pos
 * @param offset
 * @param weights
 * @param nbSynapses
 */
SimpleNeuron::SimpleNeuron(size_t index, size_t layer, NeuronConfig &conf, Position pos, Position offset, WeightMatrix &weights, WeightMatrix &localInhibWeights, size_t nbSynapses, double &outsideLat) :
        Neuron(index, layer, conf, pos, offset),
        m_events(boost::circular_buffer<Event>(1000)),
        m_topDownInhibitionEvents(boost::circular_buffer<NeuronEvent>(1000)),
        m_lateralInhibitionEvents(boost::circular_buffer<NeuronEvent>(1000)),
        m_topDownExcitationEvents(boost::circular_buffer<NeuronEvent>(1000)),
        m_lateralExcitationEvents(boost::circular_buffer<NeuronEvent>(1000)),
        m_waitingList(std::priority_queue<Event, std::vector<Event>, CompareEventsTimestamp>()),
        m_sharedWeights(weights),
        m_sharedInhibWeights(localInhibWeights),
        m_sizeInit(0),
        m_outsideLatNorm(outsideLat)
    {
        for (size_t synapse = 0; synapse < nbSynapses; synapse++) {
            m_delays.push_back(static_cast<size_t>(synapse * conf.SYNAPSE_DELAY));
        }
        m_pliLTP = m_conf.ETA_ILTP * m_conf.LATERAL_NORM_FACTOR;
        m_pliLTD = m_conf.ETA_ILTD * m_conf.LATERAL_NORM_FACTOR;
        m_ptiLTP = m_conf.ETA_ILTP * m_conf.TOPDOWN_NORM_FACTOR;
        m_ptiLTD = m_conf.ETA_ILTD * m_conf.TOPDOWN_NORM_FACTOR;
}

/**
 * Updates neuron internal state after the arrival of an event
 * Checks first if there is some synaptic delays defined in the network.
 * @param event
 * @return
 */
inline bool SimpleNeuron::newEvent(Event event) {
        m_events.push_back(event);
        return membraneUpdate(event);
}

/**
 *
 * @param event
 */
void SimpleNeuron::newLocalInhibitoryEvent(NeuronEvent event) {
    m_lateralLocalInhibitionEvents.push_back(event);
    potentialDecay(event.timestamp());
    setLastBeforeInhibitionPotential();
    adaptationPotentialDecay(event.timestamp());
    m_potential -= (m_sharedInhibWeights.get(event.id()) + refractoryPotential(event.timestamp()) + m_adaptationPotential); 
    checkNegativeLimits();
    m_timestampLastEvent = event.timestamp();
}

/**
 *
 * @param event
 */
void SimpleNeuron::newTopDownInhibitoryEvent(NeuronEvent event) {
    m_topDownInhibitionEvents.push_back(event);
    potentialDecay(event.timestamp());
    adaptationPotentialDecay(event.timestamp());
    setLastBeforeInhibitionPotential();
    m_potential -= (m_topDownInhibitionWeights.at(event.id()) + refractoryPotential(event.timestamp()) + m_adaptationPotential);
    m_timestampLastEvent = event.timestamp();
    checkNegativeLimits();
}

/**
 *
 * @param event
 */
bool SimpleNeuron::newTopDownExcitatoryEvent(NeuronEvent event) {
    m_topDownExcitationEvents.push_back(event);
    potentialDecay(event.timestamp());
    adaptationPotentialDecay(event.timestamp());
    setLastBeforeInhibitionPotential();
    m_potential += (m_topDownExcitationWeights.at(event.id()) - refractoryPotential(event.timestamp()) - m_adaptationPotential);
    m_timestampLastEvent = event.timestamp();
    checkNegativeLimits();
    if (m_potential > m_threshold) {
        spike(event.timestamp());
        return true;
    }
    return false;
}

/**
 *
 * @param event
 */
void SimpleNeuron::newLateralInhibitoryEvent(NeuronEvent event) {
    m_lateralInhibitionEvents.push_back(event);
    potentialDecay(event.timestamp());
    adaptationPotentialDecay(event.timestamp());
    setLastBeforeInhibitionPotential();
    if(!m_noLat) {
        m_potential -= (m_lateralInhibitionWeights.at(event.id()) + refractoryPotential(event.timestamp()) + m_adaptationPotential);
    }
    m_timestampLastEvent = event.timestamp();
    checkNegativeLimits();
}

/**
 *
 * @param event
 */
bool SimpleNeuron::newLateralExcitatoryEvent(NeuronEvent event) {
    m_lateralExcitationEvents.push_back(event);
    potentialDecay(event.timestamp());
    adaptationPotentialDecay(event.timestamp());
    setLastBeforeInhibitionPotential();
    m_potential += (m_lateralExcitationWeights.at(event.id()) - refractoryPotential(event.timestamp()) - - m_adaptationPotential);
    m_timestampLastEvent = event.timestamp();
    checkNegativeLimits();
    if (m_potential > m_threshold) {
        spike(event.timestamp());
        return true;
    }
    return false;
}

/**
 * Updates the membrane potential using the newly arrived event.
 * Updates some homeostatic mechanisms such as the refractory period, potential decay and spike rate adaptation.
 * If the membrane potential exceeds the threshold, the neuron spikes.
 * @param event
 * @return
 */
inline bool SimpleNeuron::membraneUpdate(Event event) {
    if(m_firstInput==0) {
        m_firstInput = event.timestamp();
    }
    potentialDecay(event.timestamp());
    adaptationPotentialDecay(event.timestamp());
    m_potential += m_sharedWeights.get(event.polarity(), event.camera(), event.synapse(), event.x(), event.y())
                        - refractoryPotential(event.timestamp()) - m_adaptationPotential;
    m_timestampLastEvent = event.timestamp();
    checkNegativeLimits();
    if (m_potential > m_threshold) {
        spike(event.timestamp());
        m_spikeCounter+=1;
        m_sizeInit+=1;
        updateTimeSurface(event.timestamp(), event.polarity());
        return true;
    }
    return false;
}

/**
 * Updates the spike timings and spike counters.
 * Also increases the secondary membrane potential use in the spike rate adaptation mechanism.
 * @param time
 */
inline void SimpleNeuron::spike(const size_t time) {
    m_lastSpikingTime = m_spikingTime;
    m_spikingTime = time;
    m_spike = true;
    ++m_spikeRateCounter;
    ++m_totalSpike;
    m_spikingPotential = m_potential;
    m_potential = m_conf.VRESET;
    spikeRateAdaptation();
     if (m_conf.TRACKING == "partial") {
        // m_trackingSpikeTrain.push_back(time - m_firstInput);
        m_trackingSpikeTrain.push_back(time);
    }
}

/**
 * Updates the synaptic weights using the STDP learning rule.
 * Only the synapses from which events arrived are updated.
 * Normalizes the weights after the update.
 */
inline void SimpleNeuron::weightUpdate() {
    if (m_conf.STDP_LEARNING == "excitatory" || m_conf.STDP_LEARNING == "all") {
        if(m_events.size() >= 0) {
            double center_lr = 50;
            double sigma_lr = 4;
            double lr_factor = 1; 
            double max_eWeight = 3; 
            double factor_eBound = 0.33; 

            if(true) {
                for (Event &event: m_events) {
                    double wj = m_sharedWeights.get(event.polarity(), event.camera(), event.synapse(), event.x(), event.y());
                    double softboundLTP = (max_eWeight - wj) * factor_eBound; 
                    double dLTP = softboundLTP * m_decay * m_eLTP * lr_factor * exp(-static_cast<double>(m_spikingTime - event.timestamp()) / m_conf.TAU_LTP);    
                    double softboundLTD = (wj) * factor_eBound;      
                        m_sharedWeights.get(event.polarity(), event.camera(), event.synapse(), event.x(), event.y()) += dLTP;
                    if(m_lastSpikingTime != 0) {
                            double dLTD = softboundLTD * m_eLTD * lr_factor * exp(-static_cast<double>(event.timestamp() - m_lastSpikingTime) / m_conf.TAU_LTD);
                            m_sharedWeights.get(event.polarity(), event.camera(), event.synapse(), event.x(), event.y()) += dLTD;
                    }
                    
                    std::vector<size_t> dim = m_sharedWeights.getDimensions();
                    bool a = event.polarity();
                    uint16_t b = event.camera();
                    uint16_t c = event.synapse();
                    uint16_t d = event.x(); 
                    uint16_t e = event.y(); 
                    if (m_sharedWeights.get(event.polarity(), event.camera(), event.synapse(), event.x(), event.y()) < 0) {
                        m_sharedWeights.get(event.polarity(), event.camera(), event.synapse(), event.x(), event.y()) = 0;
                    }
                }
            }
        }
        m_sharedWeights.normalizeL1Norm(m_conf.NORM_FACTOR);
        double max_iWeight = 50;
        double factor_iBound = 0.02;
        for (NeuronEvent &event: m_lateralLocalInhibitionEvents) {
            double wj = m_sharedInhibWeights.get(event.id());
            double softboundLTP = (max_iWeight - wj) * factor_iBound; 
            double dLTP = softboundLTP * m_iLTP * exp(-static_cast<double>(m_spikingTime - event.timestamp()) / m_conf.TAU_LTP);
            double softboundLTD = (wj) * factor_iBound;    
            m_sharedInhibWeights.get(event.id()) += dLTP;
                    
            if(m_lastSpikingTime != 0) {
                double dLTD = softboundLTD * m_iLTD * exp(-static_cast<double>(event.timestamp() - m_lastSpikingTime) / m_conf.TAU_LTD);
                m_sharedInhibWeights.get(event.id()) += dLTD;
            }
            
            if (m_sharedInhibWeights.get(event.id()) < 0) {
                m_sharedInhibWeights.get(event.id()) = 0;
            }  
        }
        m_sharedInhibWeights.normalizeL1Norm(m_conf.ETA_INH);
    }
    
    if ((m_conf.STDP_LEARNING == "inhibitory" || m_conf.STDP_LEARNING == "all") && !m_noLat) {
        double max_tdiWeight = 50;
        double factor_tdiBound = 0.02;
        for (NeuronEvent &event: m_topDownInhibitionEvents) {
            double wj = m_topDownInhibitionWeights.at(event.id());
            double softboundLTP = (max_tdiWeight - wj) * factor_tdiBound; 
            double dLTP = softboundLTP * m_ptiLTP * exp(-static_cast<double>(m_spikingTime - event.timestamp()) / m_conf.TAU_LTP);
            double softboundLTD = (wj) * factor_tdiBound; 
            m_topDownInhibitionWeights.at(event.id()) += dLTP;
                     
            if(m_lastSpikingTime!=0) {
                double dLTD = softboundLTD * m_ptiLTD * exp(-static_cast<double>(event.timestamp() - m_lastSpikingTime) / m_conf.TAU_LTD);
                m_topDownInhibitionWeights.at(event.id()) += dLTD;
            }
            if (m_topDownInhibitionWeights.at(event.id()) < 0) {
                m_topDownInhibitionWeights.at(event.id()) = 0;
            }
        }
        m_topDownInhibitionWeights.normalizeL1Norm(m_conf.TOPDOWN_NORM_FACTOR);

        double max_liWeight = 50;
        double factor_liBound = 0.02;
        if(m_lateralInhibitionWeights.getSize() == 1) {
            for (auto neur: m_lateralDynamicInhibitionConnections) {
                m_outsideLatNorm -= m_lateralInhibitionWeights.at(neur.get().getIndex());
            }
        }
        for (NeuronEvent &event: m_lateralInhibitionEvents) {
            double wj = m_lateralInhibitionWeights.at(event.id());
            double softboundLTP = (max_liWeight - wj) * factor_liBound; 
            double dLTP = softboundLTP * m_pliLTP * exp(-static_cast<double>(m_spikingTime - event.timestamp()) / m_conf.TAU_LTP);
            double softboundLTD =  (wj) * factor_liBound; 
            m_lateralInhibitionWeights.at(event.id()) += dLTP;
                    
            if(m_lastSpikingTime != 0) {
                double dLTD = softboundLTD * m_pliLTD * exp(-static_cast<double>(event.timestamp() - m_lastSpikingTime) / m_conf.TAU_LTD);
                m_lateralInhibitionWeights.at(event.id()) += dLTD;
            }

            if (m_lateralInhibitionWeights.at(event.id()) < 0) {
                m_lateralInhibitionWeights.at(event.id()) = 0;
            }
        }
        if(m_lateralInhibitionWeights.getSize() > 1) {
            m_lateralInhibitionWeights.normalizeL1Norm(m_conf.LATERAL_NORM_FACTOR);
        }
        for (auto neur: m_lateralDynamicInhibitionConnections) {
            m_outsideLatNorm += m_lateralInhibitionWeights.at(neur.get().getIndex());
            // m_lateralInhibitionWeights.at(neur.get().getIndex()) *= m_conf.LATERAL_NORM_FACTOR / m_outsideLatNorm;
        }
    }
    m_events.clear();
    m_topDownInhibitionEvents.clear();
    m_lateralLocalInhibitionEvents.clear();
    m_lateralInhibitionEvents.clear();
}

void SimpleNeuron::resetSimpleNeuron() {
    m_spikeCounter = 0;
    m_events.clear();
    m_topDownInhibitionEvents.clear();
    m_lateralInhibitionEvents.clear();
    m_lateralLocalInhibitionEvents.clear();
    m_sizeInit=0;
}

void SimpleNeuron::normalizeL1Weights() {
    m_sharedWeights.normalizeL1Norm(m_conf.NORM_FACTOR);
    m_sharedInhibWeights.normalizeL1Norm(m_conf.ETA_INH);
    if(m_lateralInhibitionWeights.getSize() > 1) {
        m_lateralInhibitionWeights.normalizeL1Norm(m_conf.LATERAL_NORM_FACTOR);
    }
    else {
        for (auto neur: m_lateralDynamicInhibitionConnections) {
            m_lateralInhibitionWeights.at(neur.get().getIndex()) *= m_conf.LATERAL_NORM_FACTOR / m_outsideLatNorm;
        }
    }
    m_topDownInhibitionWeights.normalizeL1Norm(m_conf.TOPDOWN_NORM_FACTOR);

}

/**
 *
 * @param filePath
 */
void SimpleNeuron::saveWeights(const std::string &filePath) {
    auto weightsFile = filePath + std::to_string(m_index);
    m_sharedWeights.saveWeightsToNumpyFile(weightsFile);
}

/**
 *
 * @param filePath
 */
void SimpleNeuron::savePlasticLocalInhibitionWeights(const std::string &filePath) {
    auto weightsFile = filePath + std::to_string(m_index) + "lli"; 
    m_sharedInhibWeights.saveWeightsToNumpyFile(weightsFile);
}

/**
 *
 * @param filePath
 */
void SimpleNeuron::saveLateralInhibitionWeights(std::string &filePath) {
    auto weightsFile = filePath + std::to_string(m_index) + "li";
    m_lateralInhibitionWeights.saveWeightsToNumpyFile(weightsFile);
}

/**
 *
 * @param filePath
 */
void SimpleNeuron::saveLateralExcitationWeights(std::string &filePath) {
    auto weightsFile = filePath + std::to_string(m_index) + "le";
    m_lateralExcitationWeights.saveWeightsToNumpyFile(weightsFile);
}

/**
 *
 * @param filePath
 */
void SimpleNeuron::saveTopDownInhibitionWeights(std::string &filePath) {
    auto weightsFile = filePath + std::to_string(m_index) + "tdi";
    m_topDownInhibitionWeights.saveWeightsToNumpyFile(weightsFile);
}

/**
 *
 * @param filePath
 */
void SimpleNeuron::saveTopDownExcitationWeights(std::string &filePath) {
    auto weightsFile = filePath + std::to_string(m_index) + "tde";
    m_topDownExcitationWeights.saveWeightsToNumpyFile(weightsFile);
}

/**
 *
 * @param filePath
 */
void SimpleNeuron::loadWeights(std::string &filePath) {
    auto numpyFile = filePath + std::to_string(m_index) + ".npy";
    m_sharedWeights.loadNumpyFile(numpyFile);
}

/**
 *
 * @param arrayNPZ
 */
void SimpleNeuron::loadWeights(cnpy::npz_t &arrayNPZ) {
    auto arrayName = std::to_string(m_index);
    m_sharedWeights.loadNumpyFile(arrayNPZ, arrayName);
}

/**
 *
 * @param filePath
 */
void SimpleNeuron::loadPlasticLocalInhibitionWeights(std::string &filePath) {
    auto numpyFile = filePath + std::to_string(m_index) + "lli.npy";
    m_sharedInhibWeights.loadNumpyFile(numpyFile);
}

/**
 *
 * @param arrayNPZ
 */
void SimpleNeuron::loadPlasticLocalInhibitionWeights(cnpy::npz_t &arrayNPZ) {
    auto arrayName = std::to_string(m_index) + "lli";
    m_sharedInhibWeights.loadNumpyFile(arrayNPZ, arrayName);
}

/**
 *
 * @param arrayNPZ
 */
void SimpleNeuron::loadLateralInhibitionWeights(cnpy::npz_t &arrayNPZ) {
    auto arrayName = std::to_string(m_index);
    m_lateralInhibitionWeights.loadNumpyFile(arrayNPZ, arrayName);
}

/**
 *
 * @param arrayNPZ
 */
void SimpleNeuron::loadLateralExcitationWeights(cnpy::npz_t &arrayNPZ) {
    auto arrayName = std::to_string(m_index);
    m_lateralInhibitionWeights.loadNumpyFile(arrayNPZ, arrayName);
}

/**
 *
 * @param filePath
 */
void SimpleNeuron::loadLateralInhibitionWeights(std::string &filePath) {
    auto numpyFile = filePath + std::to_string(m_index) + "li.npy";
    m_lateralInhibitionWeights.loadNumpyFile(numpyFile);
}

/**
 *
 * @param filePath
 */
void SimpleNeuron::loadLateralExcitationWeights(std::string &filePath) {
    auto numpyFile = filePath + std::to_string(m_index) + "le.npy";
    m_lateralExcitationWeights.loadNumpyFile(numpyFile);
}

/**
 *
 * @param arrayNPZ
 */
void SimpleNeuron::loadTopDownInhibitionWeights(cnpy::npz_t &arrayNPZ) {
    auto arrayName = std::to_string(m_index);
    m_topDownInhibitionWeights.loadNumpyFile(arrayNPZ, arrayName);
}

/**
 *
 * @param arrayNPZ
 */
void SimpleNeuron::loadTopDownExcitationWeights(cnpy::npz_t &arrayNPZ) {
    auto arrayName = std::to_string(m_index);
    m_topDownExcitationWeights.loadNumpyFile(arrayNPZ, arrayName);
}

/**
 *
 * @param filePath
 */
void SimpleNeuron::loadTopDownInhibitionWeights(std::string &filePath) {
    auto numpyFile = filePath + std::to_string(m_index) + "tdi.npy";
    m_topDownInhibitionWeights.loadNumpyFile(numpyFile);
}

/**
 *
 * @param filePath
 */
void SimpleNeuron::loadTopDownExcitationWeights(std::string &filePath) {
    auto numpyFile = filePath + std::to_string(m_index) + "tde.npy";
    m_topDownExcitationWeights.loadNumpyFile(numpyFile);
}

/**
 *
 * @return
 */
WeightMatrix &SimpleNeuron::getWeightsMatrix() {
    return m_sharedWeights;
}

double SimpleNeuron::getWeightsMatrixNorm() {
    return m_sharedWeights.getNorm();
}

/**
 *
 * @return
 */
std::vector<size_t> SimpleNeuron::getWeightsDimension() {
    return m_sharedWeights.getDimensions();
}

void SimpleNeuron::updateTimeSurface(double time, bool polarity) {
    m_outputs.push_back(time);
}

double SimpleNeuron::getTimeSurfaceBins(double t, double tf) {
    float timeSurfaceLocal = 0;
    for (auto & ts : m_outputs) {
        if(ts >= t && ts < tf) {
            timeSurfaceLocal += 1;
        }
    }

    return timeSurfaceLocal; 
}