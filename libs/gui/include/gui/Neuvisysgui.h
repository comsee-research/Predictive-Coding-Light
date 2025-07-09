//
// Created by Thomas on 14/04/2021.
//

#ifndef NEUVISYSGUI_H
#define NEUVISYSGUI_H

#include <QMainWindow>
#include <QApplication>
#include <QFile>
#include <QFileDialog>
#include <QTextStream>
#include <QStandardItemModel>
#include <QMessageBox>
#include <QtCharts>

#include "Neuvisysthread.h"

QT_BEGIN_NAMESPACE
namespace Ui { class NeuvisysGUI; }
QT_END_NAMESPACE

class NeuvisysGUI : public QMainWindow {
Q_OBJECT

public:
    NeuvisysGUI(int argc, char **argv, const std::string &eventPath = "", const std::string &networkPath = "", QWidget *parent = nullptr);

    ~NeuvisysGUI() override;

protected:
    NeuvisysThread neuvisysThread;
    Ui::NeuvisysGUI *ui;
    QChart *eventRateChart;
    QChart *networkRateChart;
    QLineSeries *eventRateSeries;
    QLineSeries *networkRateSeries;
    QLineSeries *potentialSeries;
    QChart *potentialChart;
    QScatterSeries *spikeSeries;
    QChart *spikeChart;
    QLineSeries *valueSeries;
    QLineSeries *valueDotSeries;
    QLineSeries *tdSeries;
    QButtonGroup *buttonSelectionGroup;

    QImage m_leftImage;
    QImage m_rightImage;
    size_t m_vfWidth{};
    size_t m_vfHeight{};

    size_t m_id{};
    size_t m_zcell{};
    size_t m_camera{};
    size_t m_synapse{};
    size_t m_layer{};
    size_t precisionEvent{};
    size_t precisionPotential{};
    size_t rangePotential{};
    size_t rangeSpiketrain{};

private:
    void openConfigFiles();

    QString readConfFile(QString &directory);

    static void modifyConfFile(QString &directory, QString &text);

public slots:

    void onDisplayProgress(int progress, double time);

    void onDisplayStatistics(const std::vector<double> &eventRateTrain, const std::vector<double> &networkRateTrain);

    void onDisplayEvents(const cv::Mat &leftEventDisplay, const cv::Mat &rightEventDisplay);

    void onDisplayWeights(const std::map<size_t, cv::Mat> &weightDisplay, size_t layerViz);

    void onDisplayPotential(double spikeRate, double vreset, double threshold, const std::vector<std::pair<double, size_t>> &potentialTrain);

    void onDisplaySpike(const std::vector<std::reference_wrapper<const std::vector<size_t>>> &spikeTrains, double time);

    void onNetworkConfiguration(const std::string &sharingType, const std::vector<std::vector<size_t>> &layerPatches,
                                const std::vector<size_t> &layerSizes, const
                                std::vector<size_t> &neuronSizes);

    void onNetworkCreation(size_t nbCameras, size_t nbSynapses, const std::vector<size_t> &networkStructure, size_t vfWidth, size_t vfHeight);

    void onNetworkDestruction();

    void onConsoleMessage(const std::string &msg);

signals:

    void tabVizChanged(size_t index);

    void indexChanged(size_t index);

    void zcellChanged(size_t zcell);

    void cameraChanged(size_t camera);

    void synapseChanged(size_t synapse);

    void precisionEventChanged(size_t precisionEvent);

    void rangePotentialChanged(size_t rangePotential);

    void precisionPotentialChanged(size_t precisionPotential);

    void rangeSpikeTrainChanged(size_t rangeSpiketrain);

    void layerChanged(size_t layer);

    void stopNetwork();

    void createNetwork(std::string fileName);

private slots:

    void on_button_event_file_clicked();

    void on_button_network_directory_clicked();

    void on_button_create_network_clicked();

    void on_button_launch_network_clicked();

    void on_text_network_config_textChanged();

    void on_text_simple_cell_config_textChanged();

    void on_text_complex_cell_config_textChanged();

    void on_text_network_directory_textChanged();

    void on_button_selection_clicked(int index);

    void on_tab_visualization_currentChanged(int index);

    void on_spin_zcell_selection_valueChanged(int arg1);

    void on_spin_camera_selection_valueChanged(int arg1);

    void on_spin_synapse_selection_valueChanged(int arg1);

    void on_slider_precision_event_sliderMoved(int position);

    void on_slider_range_potential_sliderMoved(int position);

    void on_slider_precision_potential_sliderMoved(int position);

    void on_slider_range_spiketrain_sliderMoved(int position);

    void on_slider_layer_sliderMoved(int position);

    void on_button_stop_network_clicked();

};

#endif // NEUVISYSGUI_H
