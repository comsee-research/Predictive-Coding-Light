//
// Created by Thomas on 02/12/2021.
//

#include "GLWidget.h"

GLWidget::GLWidget(QWidget *parent) : QOpenGLWidget(parent) {
    m_image = QImage("");
}

void GLWidget::initializeGL() {
    // Set up the rendering context, load shaders and other resources, etc.:
    initializeOpenGLFunctions();

//    glEnable(GL_DEPTH_TEST);
//    glEnable(GL_CULL_FACE);
}

void GLWidget::resizeGL(int w, int h) {
    int side = qMin(w, h);
    glViewport((w - side) / 2, (h - side) / 2, side, side);
}

void GLWidget::paintGL() {
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

    QPainter painter;
    painter.begin(this);
    painter.setRenderHint(QPainter::Antialiasing);
    painter.drawImage(0, 0, m_image);
    painter.end();

//    m_texture = new QOpenGLTexture(m_image.mirrored());
//    m_texture->bind();
//    glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
}

