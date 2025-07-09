/********************************************************************************
** Form generated from reading UI file 'neuvisysgui.ui'
**
** Created by: Qt User Interface Compiler version 5.12.8
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_NEUVISYSGUI_H
#define UI_NEUVISYSGUI_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QFormLayout>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QGroupBox>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QLCDNumber>
#include <QtWidgets/QLabel>
#include <QtWidgets/QLineEdit>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QMenu>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QProgressBar>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QRadioButton>
#include <QtWidgets/QScrollArea>
#include <QtWidgets/QSlider>
#include <QtWidgets/QSpinBox>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QTabWidget>
#include <QtWidgets/QTextEdit>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>
#include "GLWidget.h"
#include "QtCharts"

QT_BEGIN_NAMESPACE

class Ui_NeuvisysGUI
{
public:
    QWidget *centralwidget;
    QGridLayout *gridLayout_9;
    QFormLayout *formLayoutProgress;
    QLCDNumber *lcd_sim_time;
    QProgressBar *progressBar;
    QTabWidget *tab_selection;
    QWidget *launch_options;
    QGridLayout *gridLayout_2;
    QVBoxLayout *verticalLayoutNetwork;
    QPushButton *button_launch_network;
    QPushButton *button_stop_network;
    QTextEdit *console;
    QGridLayout *gridLayoutFiles;
    QPushButton *button_event_file;
    QLineEdit *text_event_file;
    QSpinBox *number_runs;
    QLabel *label_number_runs;
    QHBoxLayout *BoxLayoutChoice;
    QRadioButton *recording;
    QRadioButton *realtime;
    QRadioButton *simulation;
    QRadioButton *events_only;
    QWidget *selection;
    QVBoxLayout *verticalLayout;
    QGroupBox *groupBoxNetworkSelection;
    QVBoxLayout *verticalLayout_4;
    QFormLayout *formLayoutNetworkSelection;
    QGroupBox *groupBoxLayerSelection;
    QGridLayout *gridLayout_19;
    QSlider *slider_layer;
    QLabel *label_camera;
    QSpinBox *spin_camera_selection;
    QLabel *label_synapse;
    QSpinBox *spin_synapse_selection;
    QGroupBox *groupBoxNeuronSelection;
    QVBoxLayout *verticalLayout_2;
    QFormLayout *formLayoutZSelection;
    QLabel *label_zcell;
    QSpinBox *spin_zcell_selection;
    QScrollArea *scrollAreaNeuronSelection;
    QWidget *scrollAreaWidgetContentsNeuronSelection;
    QGridLayout *gridLayout_24;
    QGridLayout *gridSelection;
    QGroupBox *groupBoxGraphs;
    QVBoxLayout *verticalLayout_3;
    QLabel *label_5;
    QGridLayout *gridLayoutEventViz;
    QLabel *label_precision_event;
    QLCDNumber *lcd_precision_event;
    QSlider *slider_precision_event;
    QLabel *label_potential;
    QGridLayout *gridLayoutPotential;
    QLabel *label_range_potential;
    QSlider *slider_range_potential;
    QLCDNumber *lcd_range_potential;
    QLabel *label_precision_potential;
    QSlider *slider_precision_potential;
    QLCDNumber *lcd_precision_potential;
    QLabel *label_spiketrain;
    QGridLayout *gridLayoutSpiketrain;
    QLabel *label_range_spiketrain;
    QSlider *slider_range_spiketrain;
    QLCDNumber *lcd_range_spiketrain;
    QWidget *tab_creation;
    QVBoxLayout *verticalLayout_5;
    QPushButton *button_create_network;
    QPushButton *button_network_directory;
    QLineEdit *text_network_directory;
    QTabWidget *tab_configs;
    QWidget *network_config;
    QGridLayout *gridLayout_11;
    QTextEdit *text_network_config;
    QWidget *simple_cell_config;
    QGridLayout *gridLayout_12;
    QTextEdit *text_simple_cell_config;
    QWidget *complex_cell_config;
    QGridLayout *gridLayout_13;
    QTextEdit *text_complex_cell_config;
    QTabWidget *tab_visualization;
    QWidget *event_viz;
    QGridLayout *gridLayout_6;
    QScrollArea *scroll_event_viz;
    QWidget *scrollAreaWidgetContents;
    QHBoxLayout *horizontalLayout;
    GLWidget *opengl_left_events;
    GLWidget *opengl_right_events;
    QWidget *statistics;
    QGridLayout *gridLayout_3;
    QChartView *eventRateView;
    QChartView *networkRateView;
    QWidget *weights;
    QGridLayout *gridLayout_7;
    QScrollArea *scrollAreaWeights;
    QWidget *scrollAreaLayoutWeights;
    QGridLayout *gridLayout_15;
    QGridLayout *weightLayout;
    QWidget *potential;
    QGridLayout *gridLayout_4;
    QHBoxLayout *neuron_statistics;
    QLabel *label_spike_rate;
    QLCDNumber *lcd_spike_rate;
    QLabel *label_threshold;
    QLCDNumber *lcd_threshold;
    QLabel *label_reset;
    QLCDNumber *lcd_reset;
    QChartView *potentialView;
    QWidget *spiketrain;
    QGridLayout *gridLayout_10;
    QChartView *spikeView;
    QWidget *reward;
    QGridLayout *gridLayout_20;
    QChartView *rewardView;
    QWidget *action;
    QGridLayout *gridLayout;
    QChartView *actionView;
    QStatusBar *statusbar;
    QMenuBar *menubar;
    QMenu *menuNeuvisys_gui;
    QButtonGroup *modeChoice;

    void setupUi(QMainWindow *NeuvisysGUI)
    {
        if (NeuvisysGUI->objectName().isEmpty())
            NeuvisysGUI->setObjectName(QString::fromUtf8("NeuvisysGUI"));
        NeuvisysGUI->resize(1402, 776);
        centralwidget = new QWidget(NeuvisysGUI);
        centralwidget->setObjectName(QString::fromUtf8("centralwidget"));
        gridLayout_9 = new QGridLayout(centralwidget);
        gridLayout_9->setObjectName(QString::fromUtf8("gridLayout_9"));
        formLayoutProgress = new QFormLayout();
        formLayoutProgress->setObjectName(QString::fromUtf8("formLayoutProgress"));
        lcd_sim_time = new QLCDNumber(centralwidget);
        lcd_sim_time->setObjectName(QString::fromUtf8("lcd_sim_time"));
        QSizePolicy sizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(lcd_sim_time->sizePolicy().hasHeightForWidth());
        lcd_sim_time->setSizePolicy(sizePolicy);
        lcd_sim_time->setMinimumSize(QSize(200, 30));
        lcd_sim_time->setSmallDecimalPoint(false);
        lcd_sim_time->setDigitCount(15);

        formLayoutProgress->setWidget(0, QFormLayout::LabelRole, lcd_sim_time);

        progressBar = new QProgressBar(centralwidget);
        progressBar->setObjectName(QString::fromUtf8("progressBar"));
        progressBar->setMinimumSize(QSize(0, 30));
        progressBar->setValue(24);

        formLayoutProgress->setWidget(0, QFormLayout::FieldRole, progressBar);


        gridLayout_9->addLayout(formLayoutProgress, 3, 0, 1, 2);

        tab_selection = new QTabWidget(centralwidget);
        tab_selection->setObjectName(QString::fromUtf8("tab_selection"));
        QSizePolicy sizePolicy1(QSizePolicy::Minimum, QSizePolicy::Expanding);
        sizePolicy1.setHorizontalStretch(0);
        sizePolicy1.setVerticalStretch(0);
        sizePolicy1.setHeightForWidth(tab_selection->sizePolicy().hasHeightForWidth());
        tab_selection->setSizePolicy(sizePolicy1);
        tab_selection->setMinimumSize(QSize(500, 0));
        tab_selection->setMaximumSize(QSize(500, 16777215));
        launch_options = new QWidget();
        launch_options->setObjectName(QString::fromUtf8("launch_options"));
        QSizePolicy sizePolicy2(QSizePolicy::Preferred, QSizePolicy::Preferred);
        sizePolicy2.setHorizontalStretch(0);
        sizePolicy2.setVerticalStretch(0);
        sizePolicy2.setHeightForWidth(launch_options->sizePolicy().hasHeightForWidth());
        launch_options->setSizePolicy(sizePolicy2);
        gridLayout_2 = new QGridLayout(launch_options);
        gridLayout_2->setObjectName(QString::fromUtf8("gridLayout_2"));
        verticalLayoutNetwork = new QVBoxLayout();
        verticalLayoutNetwork->setObjectName(QString::fromUtf8("verticalLayoutNetwork"));
        button_launch_network = new QPushButton(launch_options);
        button_launch_network->setObjectName(QString::fromUtf8("button_launch_network"));

        verticalLayoutNetwork->addWidget(button_launch_network);

        button_stop_network = new QPushButton(launch_options);
        button_stop_network->setObjectName(QString::fromUtf8("button_stop_network"));

        verticalLayoutNetwork->addWidget(button_stop_network);

        console = new QTextEdit(launch_options);
        console->setObjectName(QString::fromUtf8("console"));
        console->setReadOnly(true);

        verticalLayoutNetwork->addWidget(console);


        gridLayout_2->addLayout(verticalLayoutNetwork, 3, 0, 1, 1);

        gridLayoutFiles = new QGridLayout();
        gridLayoutFiles->setObjectName(QString::fromUtf8("gridLayoutFiles"));
        button_event_file = new QPushButton(launch_options);
        button_event_file->setObjectName(QString::fromUtf8("button_event_file"));

        gridLayoutFiles->addWidget(button_event_file, 0, 0, 1, 1);

        text_event_file = new QLineEdit(launch_options);
        text_event_file->setObjectName(QString::fromUtf8("text_event_file"));

        gridLayoutFiles->addWidget(text_event_file, 0, 2, 1, 1);

        number_runs = new QSpinBox(launch_options);
        number_runs->setObjectName(QString::fromUtf8("number_runs"));
        number_runs->setMaximum(10000);

        gridLayoutFiles->addWidget(number_runs, 2, 2, 1, 1);

        label_number_runs = new QLabel(launch_options);
        label_number_runs->setObjectName(QString::fromUtf8("label_number_runs"));

        gridLayoutFiles->addWidget(label_number_runs, 2, 0, 1, 1);


        gridLayout_2->addLayout(gridLayoutFiles, 1, 0, 1, 1);

        BoxLayoutChoice = new QHBoxLayout();
        BoxLayoutChoice->setObjectName(QString::fromUtf8("BoxLayoutChoice"));
        recording = new QRadioButton(launch_options);
        modeChoice = new QButtonGroup(NeuvisysGUI);
        modeChoice->setObjectName(QString::fromUtf8("modeChoice"));
        modeChoice->addButton(recording);
        recording->setObjectName(QString::fromUtf8("recording"));
        recording->setChecked(true);

        BoxLayoutChoice->addWidget(recording);

        realtime = new QRadioButton(launch_options);
        modeChoice->addButton(realtime);
        realtime->setObjectName(QString::fromUtf8("realtime"));

        BoxLayoutChoice->addWidget(realtime);

        simulation = new QRadioButton(launch_options);
        modeChoice->addButton(simulation);
        simulation->setObjectName(QString::fromUtf8("simulation"));

        BoxLayoutChoice->addWidget(simulation);

        events_only = new QRadioButton(launch_options);
        modeChoice->addButton(events_only);
        events_only->setObjectName(QString::fromUtf8("events_only"));

        BoxLayoutChoice->addWidget(events_only);


        gridLayout_2->addLayout(BoxLayoutChoice, 0, 0, 1, 1);

        tab_selection->addTab(launch_options, QString());
        selection = new QWidget();
        selection->setObjectName(QString::fromUtf8("selection"));
        sizePolicy2.setHeightForWidth(selection->sizePolicy().hasHeightForWidth());
        selection->setSizePolicy(sizePolicy2);
        verticalLayout = new QVBoxLayout(selection);
        verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
        groupBoxNetworkSelection = new QGroupBox(selection);
        groupBoxNetworkSelection->setObjectName(QString::fromUtf8("groupBoxNetworkSelection"));
        verticalLayout_4 = new QVBoxLayout(groupBoxNetworkSelection);
        verticalLayout_4->setObjectName(QString::fromUtf8("verticalLayout_4"));
        formLayoutNetworkSelection = new QFormLayout();
        formLayoutNetworkSelection->setObjectName(QString::fromUtf8("formLayoutNetworkSelection"));
        groupBoxLayerSelection = new QGroupBox(groupBoxNetworkSelection);
        groupBoxLayerSelection->setObjectName(QString::fromUtf8("groupBoxLayerSelection"));
        gridLayout_19 = new QGridLayout(groupBoxLayerSelection);
        gridLayout_19->setObjectName(QString::fromUtf8("gridLayout_19"));
        slider_layer = new QSlider(groupBoxLayerSelection);
        slider_layer->setObjectName(QString::fromUtf8("slider_layer"));
        slider_layer->setOrientation(Qt::Horizontal);
        slider_layer->setTickPosition(QSlider::TicksAbove);

        gridLayout_19->addWidget(slider_layer, 0, 0, 1, 1);


        formLayoutNetworkSelection->setWidget(0, QFormLayout::SpanningRole, groupBoxLayerSelection);

        label_camera = new QLabel(groupBoxNetworkSelection);
        label_camera->setObjectName(QString::fromUtf8("label_camera"));

        formLayoutNetworkSelection->setWidget(1, QFormLayout::LabelRole, label_camera);

        spin_camera_selection = new QSpinBox(groupBoxNetworkSelection);
        spin_camera_selection->setObjectName(QString::fromUtf8("spin_camera_selection"));

        formLayoutNetworkSelection->setWidget(1, QFormLayout::FieldRole, spin_camera_selection);

        label_synapse = new QLabel(groupBoxNetworkSelection);
        label_synapse->setObjectName(QString::fromUtf8("label_synapse"));

        formLayoutNetworkSelection->setWidget(2, QFormLayout::LabelRole, label_synapse);

        spin_synapse_selection = new QSpinBox(groupBoxNetworkSelection);
        spin_synapse_selection->setObjectName(QString::fromUtf8("spin_synapse_selection"));

        formLayoutNetworkSelection->setWidget(2, QFormLayout::FieldRole, spin_synapse_selection);


        verticalLayout_4->addLayout(formLayoutNetworkSelection);


        verticalLayout->addWidget(groupBoxNetworkSelection);

        groupBoxNeuronSelection = new QGroupBox(selection);
        groupBoxNeuronSelection->setObjectName(QString::fromUtf8("groupBoxNeuronSelection"));
        verticalLayout_2 = new QVBoxLayout(groupBoxNeuronSelection);
        verticalLayout_2->setObjectName(QString::fromUtf8("verticalLayout_2"));
        formLayoutZSelection = new QFormLayout();
        formLayoutZSelection->setObjectName(QString::fromUtf8("formLayoutZSelection"));
        label_zcell = new QLabel(groupBoxNeuronSelection);
        label_zcell->setObjectName(QString::fromUtf8("label_zcell"));

        formLayoutZSelection->setWidget(0, QFormLayout::LabelRole, label_zcell);

        spin_zcell_selection = new QSpinBox(groupBoxNeuronSelection);
        spin_zcell_selection->setObjectName(QString::fromUtf8("spin_zcell_selection"));

        formLayoutZSelection->setWidget(0, QFormLayout::FieldRole, spin_zcell_selection);


        verticalLayout_2->addLayout(formLayoutZSelection);

        scrollAreaNeuronSelection = new QScrollArea(groupBoxNeuronSelection);
        scrollAreaNeuronSelection->setObjectName(QString::fromUtf8("scrollAreaNeuronSelection"));
        scrollAreaNeuronSelection->setWidgetResizable(true);
        scrollAreaWidgetContentsNeuronSelection = new QWidget();
        scrollAreaWidgetContentsNeuronSelection->setObjectName(QString::fromUtf8("scrollAreaWidgetContentsNeuronSelection"));
        scrollAreaWidgetContentsNeuronSelection->setGeometry(QRect(0, 0, 452, 160));
        gridLayout_24 = new QGridLayout(scrollAreaWidgetContentsNeuronSelection);
        gridLayout_24->setObjectName(QString::fromUtf8("gridLayout_24"));
        gridSelection = new QGridLayout();
        gridSelection->setObjectName(QString::fromUtf8("gridSelection"));
        gridSelection->setSizeConstraint(QLayout::SetMinAndMaxSize);

        gridLayout_24->addLayout(gridSelection, 0, 0, 1, 1);

        scrollAreaNeuronSelection->setWidget(scrollAreaWidgetContentsNeuronSelection);

        verticalLayout_2->addWidget(scrollAreaNeuronSelection);


        verticalLayout->addWidget(groupBoxNeuronSelection);

        groupBoxGraphs = new QGroupBox(selection);
        groupBoxGraphs->setObjectName(QString::fromUtf8("groupBoxGraphs"));
        verticalLayout_3 = new QVBoxLayout(groupBoxGraphs);
        verticalLayout_3->setObjectName(QString::fromUtf8("verticalLayout_3"));
        label_5 = new QLabel(groupBoxGraphs);
        label_5->setObjectName(QString::fromUtf8("label_5"));

        verticalLayout_3->addWidget(label_5, 0, Qt::AlignHCenter);

        gridLayoutEventViz = new QGridLayout();
        gridLayoutEventViz->setObjectName(QString::fromUtf8("gridLayoutEventViz"));
        label_precision_event = new QLabel(groupBoxGraphs);
        label_precision_event->setObjectName(QString::fromUtf8("label_precision_event"));

        gridLayoutEventViz->addWidget(label_precision_event, 0, 0, 1, 1);

        lcd_precision_event = new QLCDNumber(groupBoxGraphs);
        lcd_precision_event->setObjectName(QString::fromUtf8("lcd_precision_event"));

        gridLayoutEventViz->addWidget(lcd_precision_event, 0, 2, 1, 1);

        slider_precision_event = new QSlider(groupBoxGraphs);
        slider_precision_event->setObjectName(QString::fromUtf8("slider_precision_event"));
        slider_precision_event->setOrientation(Qt::Horizontal);
        slider_precision_event->setTickPosition(QSlider::NoTicks);
        slider_precision_event->setTickInterval(100);

        gridLayoutEventViz->addWidget(slider_precision_event, 0, 1, 1, 1);


        verticalLayout_3->addLayout(gridLayoutEventViz);

        label_potential = new QLabel(groupBoxGraphs);
        label_potential->setObjectName(QString::fromUtf8("label_potential"));

        verticalLayout_3->addWidget(label_potential, 0, Qt::AlignHCenter);

        gridLayoutPotential = new QGridLayout();
        gridLayoutPotential->setObjectName(QString::fromUtf8("gridLayoutPotential"));
        label_range_potential = new QLabel(groupBoxGraphs);
        label_range_potential->setObjectName(QString::fromUtf8("label_range_potential"));

        gridLayoutPotential->addWidget(label_range_potential, 0, 0, 1, 1);

        slider_range_potential = new QSlider(groupBoxGraphs);
        slider_range_potential->setObjectName(QString::fromUtf8("slider_range_potential"));
        slider_range_potential->setMaximum(100000);
        slider_range_potential->setOrientation(Qt::Horizontal);
        slider_range_potential->setTickPosition(QSlider::NoTicks);
        slider_range_potential->setTickInterval(1000);

        gridLayoutPotential->addWidget(slider_range_potential, 0, 1, 1, 1);

        lcd_range_potential = new QLCDNumber(groupBoxGraphs);
        lcd_range_potential->setObjectName(QString::fromUtf8("lcd_range_potential"));

        gridLayoutPotential->addWidget(lcd_range_potential, 0, 2, 1, 1);

        label_precision_potential = new QLabel(groupBoxGraphs);
        label_precision_potential->setObjectName(QString::fromUtf8("label_precision_potential"));

        gridLayoutPotential->addWidget(label_precision_potential, 1, 0, 1, 1);

        slider_precision_potential = new QSlider(groupBoxGraphs);
        slider_precision_potential->setObjectName(QString::fromUtf8("slider_precision_potential"));
        slider_precision_potential->setOrientation(Qt::Horizontal);
        slider_precision_potential->setTickPosition(QSlider::NoTicks);
        slider_precision_potential->setTickInterval(100);

        gridLayoutPotential->addWidget(slider_precision_potential, 1, 1, 1, 1);

        lcd_precision_potential = new QLCDNumber(groupBoxGraphs);
        lcd_precision_potential->setObjectName(QString::fromUtf8("lcd_precision_potential"));

        gridLayoutPotential->addWidget(lcd_precision_potential, 1, 2, 1, 1);


        verticalLayout_3->addLayout(gridLayoutPotential);

        label_spiketrain = new QLabel(groupBoxGraphs);
        label_spiketrain->setObjectName(QString::fromUtf8("label_spiketrain"));

        verticalLayout_3->addWidget(label_spiketrain, 0, Qt::AlignHCenter);

        gridLayoutSpiketrain = new QGridLayout();
        gridLayoutSpiketrain->setObjectName(QString::fromUtf8("gridLayoutSpiketrain"));
        label_range_spiketrain = new QLabel(groupBoxGraphs);
        label_range_spiketrain->setObjectName(QString::fromUtf8("label_range_spiketrain"));

        gridLayoutSpiketrain->addWidget(label_range_spiketrain, 2, 0, 1, 1);

        slider_range_spiketrain = new QSlider(groupBoxGraphs);
        slider_range_spiketrain->setObjectName(QString::fromUtf8("slider_range_spiketrain"));
        slider_range_spiketrain->setMaximum(10000);
        slider_range_spiketrain->setOrientation(Qt::Horizontal);
        slider_range_spiketrain->setTickPosition(QSlider::NoTicks);
        slider_range_spiketrain->setTickInterval(1000);

        gridLayoutSpiketrain->addWidget(slider_range_spiketrain, 2, 1, 1, 1);

        lcd_range_spiketrain = new QLCDNumber(groupBoxGraphs);
        lcd_range_spiketrain->setObjectName(QString::fromUtf8("lcd_range_spiketrain"));

        gridLayoutSpiketrain->addWidget(lcd_range_spiketrain, 2, 2, 1, 1);


        verticalLayout_3->addLayout(gridLayoutSpiketrain);


        verticalLayout->addWidget(groupBoxGraphs);

        tab_selection->addTab(selection, QString());
        tab_creation = new QWidget();
        tab_creation->setObjectName(QString::fromUtf8("tab_creation"));
        verticalLayout_5 = new QVBoxLayout(tab_creation);
        verticalLayout_5->setObjectName(QString::fromUtf8("verticalLayout_5"));
        button_create_network = new QPushButton(tab_creation);
        button_create_network->setObjectName(QString::fromUtf8("button_create_network"));

        verticalLayout_5->addWidget(button_create_network);

        button_network_directory = new QPushButton(tab_creation);
        button_network_directory->setObjectName(QString::fromUtf8("button_network_directory"));

        verticalLayout_5->addWidget(button_network_directory);

        text_network_directory = new QLineEdit(tab_creation);
        text_network_directory->setObjectName(QString::fromUtf8("text_network_directory"));

        verticalLayout_5->addWidget(text_network_directory);

        tab_configs = new QTabWidget(tab_creation);
        tab_configs->setObjectName(QString::fromUtf8("tab_configs"));
        network_config = new QWidget();
        network_config->setObjectName(QString::fromUtf8("network_config"));
        gridLayout_11 = new QGridLayout(network_config);
        gridLayout_11->setObjectName(QString::fromUtf8("gridLayout_11"));
        text_network_config = new QTextEdit(network_config);
        text_network_config->setObjectName(QString::fromUtf8("text_network_config"));

        gridLayout_11->addWidget(text_network_config, 0, 0, 1, 1);

        tab_configs->addTab(network_config, QString());
        simple_cell_config = new QWidget();
        simple_cell_config->setObjectName(QString::fromUtf8("simple_cell_config"));
        gridLayout_12 = new QGridLayout(simple_cell_config);
        gridLayout_12->setObjectName(QString::fromUtf8("gridLayout_12"));
        text_simple_cell_config = new QTextEdit(simple_cell_config);
        text_simple_cell_config->setObjectName(QString::fromUtf8("text_simple_cell_config"));

        gridLayout_12->addWidget(text_simple_cell_config, 0, 0, 1, 1);

        tab_configs->addTab(simple_cell_config, QString());
        complex_cell_config = new QWidget();
        complex_cell_config->setObjectName(QString::fromUtf8("complex_cell_config"));
        gridLayout_13 = new QGridLayout(complex_cell_config);
        gridLayout_13->setObjectName(QString::fromUtf8("gridLayout_13"));
        text_complex_cell_config = new QTextEdit(complex_cell_config);
        text_complex_cell_config->setObjectName(QString::fromUtf8("text_complex_cell_config"));

        gridLayout_13->addWidget(text_complex_cell_config, 0, 0, 1, 1);

        tab_configs->addTab(complex_cell_config, QString());

        verticalLayout_5->addWidget(tab_configs);

        tab_selection->addTab(tab_creation, QString());

        gridLayout_9->addWidget(tab_selection, 0, 0, 1, 1);

        tab_visualization = new QTabWidget(centralwidget);
        tab_visualization->setObjectName(QString::fromUtf8("tab_visualization"));
        event_viz = new QWidget();
        event_viz->setObjectName(QString::fromUtf8("event_viz"));
        gridLayout_6 = new QGridLayout(event_viz);
        gridLayout_6->setObjectName(QString::fromUtf8("gridLayout_6"));
        scroll_event_viz = new QScrollArea(event_viz);
        scroll_event_viz->setObjectName(QString::fromUtf8("scroll_event_viz"));
        scroll_event_viz->setWidgetResizable(true);
        scrollAreaWidgetContents = new QWidget();
        scrollAreaWidgetContents->setObjectName(QString::fromUtf8("scrollAreaWidgetContents"));
        scrollAreaWidgetContents->setGeometry(QRect(0, 0, 852, 626));
        horizontalLayout = new QHBoxLayout(scrollAreaWidgetContents);
        horizontalLayout->setObjectName(QString::fromUtf8("horizontalLayout"));
        horizontalLayout->setSizeConstraint(QLayout::SetMinAndMaxSize);
        opengl_left_events = new GLWidget(scrollAreaWidgetContents);
        opengl_left_events->setObjectName(QString::fromUtf8("opengl_left_events"));

        horizontalLayout->addWidget(opengl_left_events);

        opengl_right_events = new GLWidget(scrollAreaWidgetContents);
        opengl_right_events->setObjectName(QString::fromUtf8("opengl_right_events"));

        horizontalLayout->addWidget(opengl_right_events);

        scroll_event_viz->setWidget(scrollAreaWidgetContents);

        gridLayout_6->addWidget(scroll_event_viz, 0, 0, 1, 1);

        tab_visualization->addTab(event_viz, QString());
        statistics = new QWidget();
        statistics->setObjectName(QString::fromUtf8("statistics"));
        gridLayout_3 = new QGridLayout(statistics);
        gridLayout_3->setObjectName(QString::fromUtf8("gridLayout_3"));
        eventRateView = new QChartView(statistics);
        eventRateView->setObjectName(QString::fromUtf8("eventRateView"));

        gridLayout_3->addWidget(eventRateView, 0, 0, 1, 1);

        networkRateView = new QChartView(statistics);
        networkRateView->setObjectName(QString::fromUtf8("networkRateView"));

        gridLayout_3->addWidget(networkRateView, 1, 0, 1, 1);

        tab_visualization->addTab(statistics, QString());
        weights = new QWidget();
        weights->setObjectName(QString::fromUtf8("weights"));
        gridLayout_7 = new QGridLayout(weights);
        gridLayout_7->setObjectName(QString::fromUtf8("gridLayout_7"));
        scrollAreaWeights = new QScrollArea(weights);
        scrollAreaWeights->setObjectName(QString::fromUtf8("scrollAreaWeights"));
        scrollAreaWeights->setWidgetResizable(true);
        scrollAreaLayoutWeights = new QWidget();
        scrollAreaLayoutWeights->setObjectName(QString::fromUtf8("scrollAreaLayoutWeights"));
        scrollAreaLayoutWeights->setGeometry(QRect(0, 0, 96, 26));
        gridLayout_15 = new QGridLayout(scrollAreaLayoutWeights);
        gridLayout_15->setObjectName(QString::fromUtf8("gridLayout_15"));
        weightLayout = new QGridLayout();
        weightLayout->setObjectName(QString::fromUtf8("weightLayout"));
        weightLayout->setSizeConstraint(QLayout::SetMinAndMaxSize);

        gridLayout_15->addLayout(weightLayout, 0, 0, 1, 1);

        scrollAreaWeights->setWidget(scrollAreaLayoutWeights);

        gridLayout_7->addWidget(scrollAreaWeights, 0, 0, 1, 1);

        tab_visualization->addTab(weights, QString());
        potential = new QWidget();
        potential->setObjectName(QString::fromUtf8("potential"));
        gridLayout_4 = new QGridLayout(potential);
        gridLayout_4->setObjectName(QString::fromUtf8("gridLayout_4"));
        neuron_statistics = new QHBoxLayout();
        neuron_statistics->setObjectName(QString::fromUtf8("neuron_statistics"));
        label_spike_rate = new QLabel(potential);
        label_spike_rate->setObjectName(QString::fromUtf8("label_spike_rate"));

        neuron_statistics->addWidget(label_spike_rate);

        lcd_spike_rate = new QLCDNumber(potential);
        lcd_spike_rate->setObjectName(QString::fromUtf8("lcd_spike_rate"));
        lcd_spike_rate->setMinimumSize(QSize(0, 40));
        lcd_spike_rate->setMaximumSize(QSize(200, 100));
        lcd_spike_rate->setDigitCount(5);

        neuron_statistics->addWidget(lcd_spike_rate);

        label_threshold = new QLabel(potential);
        label_threshold->setObjectName(QString::fromUtf8("label_threshold"));

        neuron_statistics->addWidget(label_threshold);

        lcd_threshold = new QLCDNumber(potential);
        lcd_threshold->setObjectName(QString::fromUtf8("lcd_threshold"));
        lcd_threshold->setMinimumSize(QSize(0, 40));
        lcd_threshold->setMaximumSize(QSize(200, 100));
        lcd_threshold->setDigitCount(5);

        neuron_statistics->addWidget(lcd_threshold);

        label_reset = new QLabel(potential);
        label_reset->setObjectName(QString::fromUtf8("label_reset"));

        neuron_statistics->addWidget(label_reset);

        lcd_reset = new QLCDNumber(potential);
        lcd_reset->setObjectName(QString::fromUtf8("lcd_reset"));
        lcd_reset->setMinimumSize(QSize(0, 40));
        lcd_reset->setMaximumSize(QSize(200, 100));

        neuron_statistics->addWidget(lcd_reset);


        gridLayout_4->addLayout(neuron_statistics, 1, 0, 1, 1);

        potentialView = new QChartView(potential);
        potentialView->setObjectName(QString::fromUtf8("potentialView"));

        gridLayout_4->addWidget(potentialView, 0, 0, 1, 1);

        tab_visualization->addTab(potential, QString());
        spiketrain = new QWidget();
        spiketrain->setObjectName(QString::fromUtf8("spiketrain"));
        gridLayout_10 = new QGridLayout(spiketrain);
        gridLayout_10->setObjectName(QString::fromUtf8("gridLayout_10"));
        spikeView = new QChartView(spiketrain);
        spikeView->setObjectName(QString::fromUtf8("spikeView"));

        gridLayout_10->addWidget(spikeView, 0, 0, 1, 1);

        tab_visualization->addTab(spiketrain, QString());
        reward = new QWidget();
        reward->setObjectName(QString::fromUtf8("reward"));
        gridLayout_20 = new QGridLayout(reward);
        gridLayout_20->setObjectName(QString::fromUtf8("gridLayout_20"));
        rewardView = new QChartView(reward);
        rewardView->setObjectName(QString::fromUtf8("rewardView"));

        gridLayout_20->addWidget(rewardView, 0, 0, 1, 1);

        tab_visualization->addTab(reward, QString());
        action = new QWidget();
        action->setObjectName(QString::fromUtf8("action"));
        gridLayout = new QGridLayout(action);
        gridLayout->setObjectName(QString::fromUtf8("gridLayout"));
        actionView = new QChartView(action);
        actionView->setObjectName(QString::fromUtf8("actionView"));

        gridLayout->addWidget(actionView, 0, 0, 1, 1);

        tab_visualization->addTab(action, QString());

        gridLayout_9->addWidget(tab_visualization, 0, 1, 1, 1);

        NeuvisysGUI->setCentralWidget(centralwidget);
        statusbar = new QStatusBar(NeuvisysGUI);
        statusbar->setObjectName(QString::fromUtf8("statusbar"));
        NeuvisysGUI->setStatusBar(statusbar);
        menubar = new QMenuBar(NeuvisysGUI);
        menubar->setObjectName(QString::fromUtf8("menubar"));
        menubar->setGeometry(QRect(0, 0, 1402, 24));
        menuNeuvisys_gui = new QMenu(menubar);
        menuNeuvisys_gui->setObjectName(QString::fromUtf8("menuNeuvisys_gui"));
        NeuvisysGUI->setMenuBar(menubar);
        QWidget::setTabOrder(tab_selection, button_event_file);
        QWidget::setTabOrder(button_event_file, text_event_file);
        QWidget::setTabOrder(text_event_file, number_runs);
        QWidget::setTabOrder(number_runs, tab_visualization);
        QWidget::setTabOrder(tab_visualization, potentialView);
        QWidget::setTabOrder(potentialView, spikeView);

        menubar->addAction(menuNeuvisys_gui->menuAction());

        retranslateUi(NeuvisysGUI);

        tab_selection->setCurrentIndex(0);
        tab_configs->setCurrentIndex(1);
        tab_visualization->setCurrentIndex(0);


        QMetaObject::connectSlotsByName(NeuvisysGUI);
    } // setupUi

    void retranslateUi(QMainWindow *NeuvisysGUI)
    {
        NeuvisysGUI->setWindowTitle(QApplication::translate("NeuvisysGUI", "NeuvisysGUI", nullptr));
        button_launch_network->setText(QApplication::translate("NeuvisysGUI", "Launch Network", nullptr));
        button_stop_network->setText(QApplication::translate("NeuvisysGUI", "Stop Network", nullptr));
        button_event_file->setText(QApplication::translate("NeuvisysGUI", "Select Event File", nullptr));
        label_number_runs->setText(QApplication::translate("NeuvisysGUI", "Number of runs", nullptr));
        recording->setText(QApplication::translate("NeuvisysGUI", "Recording", nullptr));
        realtime->setText(QApplication::translate("NeuvisysGUI", "Realtime", nullptr));
        simulation->setText(QApplication::translate("NeuvisysGUI", "Simulation", nullptr));
        events_only->setText(QApplication::translate("NeuvisysGUI", "EventsOnly", nullptr));
        tab_selection->setTabText(tab_selection->indexOf(launch_options), QApplication::translate("NeuvisysGUI", "Launch Options", nullptr));
        groupBoxNetworkSelection->setTitle(QApplication::translate("NeuvisysGUI", "Network Selection", nullptr));
        groupBoxLayerSelection->setTitle(QApplication::translate("NeuvisysGUI", "Layer selection", nullptr));
        label_camera->setText(QApplication::translate("NeuvisysGUI", "Camera", nullptr));
        label_synapse->setText(QApplication::translate("NeuvisysGUI", "Synapse", nullptr));
        groupBoxNeuronSelection->setTitle(QApplication::translate("NeuvisysGUI", "Neuron Selection", nullptr));
        label_zcell->setText(QApplication::translate("NeuvisysGUI", "Z", nullptr));
        groupBoxGraphs->setTitle(QApplication::translate("NeuvisysGUI", "Graphs parameters", nullptr));
        label_5->setText(QApplication::translate("NeuvisysGUI", "Event Viz", nullptr));
        label_precision_event->setText(QApplication::translate("NeuvisysGUI", "Precision (ms)", nullptr));
        label_potential->setText(QApplication::translate("NeuvisysGUI", "Potential", nullptr));
        label_range_potential->setText(QApplication::translate("NeuvisysGUI", "Range (ms)", nullptr));
        label_precision_potential->setText(QApplication::translate("NeuvisysGUI", "Precision (ms)", nullptr));
        label_spiketrain->setText(QApplication::translate("NeuvisysGUI", "Spiketrain", nullptr));
        label_range_spiketrain->setText(QApplication::translate("NeuvisysGUI", "Range (ms)", nullptr));
        tab_selection->setTabText(tab_selection->indexOf(selection), QApplication::translate("NeuvisysGUI", "Selection", nullptr));
        button_create_network->setText(QApplication::translate("NeuvisysGUI", "Create Network", nullptr));
        button_network_directory->setText(QApplication::translate("NeuvisysGUI", "Select Network Directory", nullptr));
        tab_configs->setTabText(tab_configs->indexOf(network_config), QApplication::translate("NeuvisysGUI", "Network config", nullptr));
        tab_configs->setTabText(tab_configs->indexOf(simple_cell_config), QApplication::translate("NeuvisysGUI", "Simple cell config", nullptr));
        tab_configs->setTabText(tab_configs->indexOf(complex_cell_config), QApplication::translate("NeuvisysGUI", "Complex cell config", nullptr));
        tab_selection->setTabText(tab_selection->indexOf(tab_creation), QApplication::translate("NeuvisysGUI", "Network Creation", nullptr));
        tab_visualization->setTabText(tab_visualization->indexOf(event_viz), QApplication::translate("NeuvisysGUI", "Event Viz", nullptr));
        tab_visualization->setTabText(tab_visualization->indexOf(statistics), QApplication::translate("NeuvisysGUI", "Statistics", nullptr));
        tab_visualization->setTabText(tab_visualization->indexOf(weights), QApplication::translate("NeuvisysGUI", "Weights", nullptr));
        label_spike_rate->setText(QApplication::translate("NeuvisysGUI", "Spike Rate (spikes/s)", nullptr));
        label_threshold->setText(QApplication::translate("NeuvisysGUI", "Threshold (mV):", nullptr));
        label_reset->setText(QApplication::translate("NeuvisysGUI", "Reset Potential (mV)", nullptr));
        tab_visualization->setTabText(tab_visualization->indexOf(potential), QApplication::translate("NeuvisysGUI", "Potential", nullptr));
        tab_visualization->setTabText(tab_visualization->indexOf(spiketrain), QApplication::translate("NeuvisysGUI", "Spiketrain", nullptr));
        tab_visualization->setTabText(tab_visualization->indexOf(reward), QApplication::translate("NeuvisysGUI", "Reward", nullptr));
        tab_visualization->setTabText(tab_visualization->indexOf(action), QApplication::translate("NeuvisysGUI", "Action", nullptr));
        menuNeuvisys_gui->setTitle(QApplication::translate("NeuvisysGUI", "Neuvisys-gui", nullptr));
    } // retranslateUi

};

namespace Ui {
    class NeuvisysGUI: public Ui_NeuvisysGUI {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_NEUVISYSGUI_H
