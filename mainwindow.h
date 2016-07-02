#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <iostream>
#include <math.h>
#include <string>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv/ml.h>
#include <QFileDialog>
#include <QString>
#include <QDebug>

using namespace std;
using namespace cv;

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

private slots:
    void on_train_button_clicked();

private:
    Ui::MainWindow *ui;
    int numberOfTrainData;
    int blockSize;

    double sigmoidFunction(double input);
};

#endif // MAINWINDOW_H
