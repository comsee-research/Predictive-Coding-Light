//
// Created by antony on 03/05/2022
//

/**
 * This is a class that works with SNNs to train sequentially data and generate simulated sequences of bars to evaluate the 
 * surround suppression phenomenon. 
 */

#ifndef NEUVISYS_DV_SURROUND_SUPPRESSION_HPP
#define NEUVISYS_DV_SURROUND_SUPPRESSION_HPP

#include <sstream>
#include <algorithm>
#include <random>
#include <cmath>

#if __GNUC__ > 8

#include <filesystem>

namespace fs = std::filesystem;
#else
#include <experimental/filesystem> //if gcc < 8
namespace fs = std::experimental::filesystem;
#endif

#include <opencv2/opencv.hpp>

#include "NetworkHandle.hpp"
#include "config/Config.hpp"

using receptivefield = struct {
    int x_minus;
    int x_plus;
    int y_minus;
    int y_plus;
};
using rfs_container = std::vector<std::vector<receptivefield>>;
using paths_container = std::vector<std::string>;
using image_matrix = cv::Mat;
using events_list = std::vector<Event>;
using events_sequences = std::vector<events_list>;
using parameters_in_events = std::vector<std::vector<size_t>>;

class SurroundSuppression {
    paths_container m_path;
    std::string m_networkPath;
    NetworkHandle &m_network;
    rfs_container m_neuronsReceptiveField;

public:
    SurroundSuppression(const std::string &networkPath, const paths_container &path, NetworkHandle &network);

    void train(const std::string &typeOfTraining, int numberOfTimes, int epochs);

    void evaluateResponsesOnStimuli();

    void evaluateResponsesOnStimuli(const std::string &eventsPath);

    void evaluateResponsesOnStimuliMultipleOrientations(const std::string &eventsPath);

    void evaluateNaturalEvents(const std::string &eventsPath);

    void tuningCurvesData(const std::string &eventsPath);

    void tuningCurvesCounterPhase(const std::string &eventsPath);

    void randomLateralInit(int norm_factor);

    void shuffleInhibition(int c);

    void classificationDescriptor(const std::vector<std::string> &vectorOfPath, const std::vector<int> &vectorOfLabels);

    void recordSpikes(const std::string &vectorOfPath);

    void evaluateResponsesSuppressionPatch(const std::string &eventsPath);

private:
    int m_time_gap;
    int m_log_threshold;
    float m_map_threshold;
    int m_n_max;
    float m_adapt_thresh_coef_shift;
    int m_timestamp_noise;

    void getReceptiveField(int layer, const int neuron_id, receptivefield &to_return);

    void generateBars(events_sequences &ev, int numberOfTypesOfBars, const std::vector<int> &lengthsOfBars, const std::vector<int> &yPositionStart,
                      int angle, int speed, int nbPass, int thickness = 1, std::string mode = "not_eval");

    void multiBars1Length(events_sequences &ev, const std::vector<int> &yPositionStart, const std::vector<int> &lengthsOfBars, int speed,
                          int num_disparities, int nbPass, const std::string &pathToImgs, int thickness, int angle);

    void oneBarMultiLengths(events_sequences &ev, const std::vector<int> &lengthsOfBars, const std::vector<int> &yPositionStart, int angle, int speed,
                            int nbPass, const std::string &pathToImgs, int thickness, std::string mode = "not_eval");

    void findCenteredBarsPosition(int middle, int value, std::vector<int> &positionStart, std::vector<int> &listOfLengths, int numberOfBars);

    void findMiddlePointOrdinate(int &middle, int angle);

    bool verifyIfSafe();

    bool verifyIfSafe(const std::string &path, int n_thickness, int n_angles, int n_directions, int n_bars, int n_simulations, bool separate_speed, int n_speeds = 3);

    bool verifyIfSafeMultipleOrientations(const std::string &path, int n_angles, int n_simulations);

    void simulationChoices(std::string &choice, int &simulation, int &n_);

    void convertFrame(image_matrix frame, image_matrix &new_frame);

    void writeEvents(events_list &events, float delta_b, float thresh, float frame_id, int x, int y, int polarity);

    void frameToEvents(int frame_id, image_matrix frame, image_matrix reference, image_matrix &threshold_map, events_list &events);

    void createEvents(const std::string &pathToFrames, events_list &events, int nbPass);

    static bool compareImgName(std::string file_1, std::string file_2);

    parameters_in_events getParameters(events_list evts, int y, int theta, int thickness, int length, int dir, int id);

    parameters_in_events findAllPossiblePositions(int theta, int center);
};

#endif //NEUVISYS_DV_SURROUND_SUPPRESSION_HPP
