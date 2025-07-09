//
// Created by Thomas on 02/12/2021.
//

#ifndef NEUVISYS_GLWIDGET_H
#define NEUVISYS_GLWIDGET_H

#include <QOpenGLWidget>
#include <QOpenGLFunctions>
#include <QOpenGLTexture>
#include <QPainter>

QT_FORWARD_DECLARE_CLASS(QOpenGLShaderProgram)
QT_FORWARD_DECLARE_CLASS(QOpenGLTexture)

class GLWidget : public QOpenGLWidget, protected QOpenGLFunctions {
private:
    QImage m_image;
    QOpenGLTexture *m_texture = nullptr;

public:
    explicit GLWidget(QWidget *parent);

    void setImage(QImage &image) { m_image = image; }

protected:
    void initializeGL() override;
    void resizeGL(int w, int h) override;
    void paintGL() override;
};

#endif //NEUVISYS_GLWIDGET_H
