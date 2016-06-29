#-------------------------------------------------
#
# Project created by QtCreator 2016-06-30T00:04:57
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = ImgComp_ANN
TEMPLATE = app

INCLUDEPATH += /usr/local/include/opencv
LIBS += -lopencv_core \
        -lopencv_highgui \
        -lopencv_imgproc \
        -lopencv_videostab \
        -lopencv_video \
        -lopencv_ts \
        -lopencv_stitching \
        -lopencv_superres \
        -lopencv_photo \
        -lopencv_objdetect \
        -lopencv_ml \
        -lopencv_calib3d \
        -lopencv_flann \
        -lopencv_features2d \
        -lopencv_imgcodecs

SOURCES += main.cpp\
        mainwindow.cpp

HEADERS  += mainwindow.h

FORMS    += mainwindow.ui
