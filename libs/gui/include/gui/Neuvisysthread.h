//
// Created by Thomas on 14/04/2021.
//

#ifndef NEUVISYSTHREAD_H
#define NEUVISYSTHREAD_H

#include <network/Event.hpp>
//QT
#include <QThread>
#include <iostream>
using namespace std;
//std
#include <chrono>
#include <random>
#include <utility>
#include <thread>

//externals
#include <json/json.hpp>

//neuvisys
#include <network/NetworkHandle.hpp>

class NeuvisysThread : public QThread {
Q_OBJECT

protected:
    int m_initArgc;
    char **m_initArgv;
    QString m_networkPath;
    QString m_events;
    size_t m_nbPass;
    size_t m_iterations;
    cv::Mat m_leftEventDisplay;
    cv::Mat m_rightEventDisplay;
    std::map<size_t, cv::Mat> m_weightDisplay;
    std::vector<std::reference_wrapper<const std::vector<size_t>>> m_spikeTrain;
    double m_eventRate{};
    size_t m_mode = 0;
    bool m_stop = false;
    bool m_change = false;
    size_t m_currentTab = 0;
    size_t m_on_count = 0, m_off_count = 0;
    double m_endTime = 0;

    double m_value = 0;

    size_t m_id = 0; // 3985;
    size_t m_zcell = 0;
    size_t m_camera = 0;
    size_t m_synapse = 0;
    size_t m_layer = 0;

    double m_displayRate = 5000; // µs
    double m_trackRate = 10000; // µs
    size_t m_rangePotential = 10000; // µs
    size_t m_rangeSpiketrain = 1000000; // µs

    int m_action = 0;
    double m_displayTime = 0, m_trackTime = 0;
    std::string m_msg;

    // void readEventsFile(NetworkHandle &network);

    // int launchReal(NetworkHandle &network);

    void eventLoop(NetworkHandle &network, const vector<Event> &events, double time);

    void launchNetwork(NetworkHandle &network);

public:
    NeuvisysThread(int argc, char **argv, QObject *parent = nullptr);

    void render(QString networkPath, QString events, size_t nbPass, size_t mode);

    bool init() { return true; }

    void run() override;

private:

    // void readEventsFile();

    // void readEventsRealTime();

    void display(NetworkHandle &network, double time);

    void addEventToDisplay(const Event &event);

    void prepareSpikes(NetworkHandle &network);

    void prepareWeights(NetworkHandle &network);

    void sensingZone(NetworkHandle &network);

public slots:

    void onTabVizChanged(size_t index);

    void onIndexChanged(size_t index);

    void onZcellChanged(size_t zcell);

    void onCameraChanged(size_t camera);

    void onSynapseChanged(size_t synapse);

    void onPrecisionEventChanged(size_t displayRate);

    void onRangePotentialChanged(size_t rangePotential);

    void onPrecisionPotentialChanged(size_t trackRate);

    void onRangeSpikeTrainChanged(size_t rangeSpiketrain);

    void onLayerChanged(size_t layer);

    void onStopNetwork();

signals:

    void displayProgress(int progress, double time);

    void displayStatistics(const std::vector<double> &eventRateTrain, const std::vector<double> &networkRateTrain);

    void displayEvents(const cv::Mat &leftEventDisplay, const cv::Mat &rightEventDisplay);

    void displayWeights(const std::map<size_t, cv::Mat> &weightDisplay, size_t layer);

    void displayPotential(double spikeRate, double vreset, double threshold, const std::vector<std::pair<double, size_t>> &potentialTrain);

    void displaySpike(const std::vector<std::reference_wrapper<const std::vector<size_t>>> &spikeTrain, double time);

    void networkConfiguration(const std::string &sharingType, const std::vector<std::vector<size_t>> &layerPatches,
                              const std::vector<size_t> &layerSizes, const std::vector<size_t> &neuronSizes);

    void
    networkCreation(size_t nbCameras, size_t nbSynapses, const std::vector<size_t> &networkStructure, const size_t vfWidth, const size_t vfHeight);

    void networkDestruction();

    void consoleMessage(const std::string &msg);
};

#endif // NEUVISYSTHREAD_H
