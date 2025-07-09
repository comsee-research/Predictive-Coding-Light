//
// Created by Thomas on 14/04/2021.
//

#include <gui/Neuvisysgui.h>

int main(int argc, char *argv[]) {
    QApplication app(argc, argv);
    NeuvisysGUI w(argc, argv,
                  "", // empty
                 "/home/comsee/PhD_Antony/data_basic_PCL_NatComms/net1c/");
    w.setFixedSize(0.8 * QDesktopWidget().availableGeometry().size());
    w.show();
    return app.exec();
}