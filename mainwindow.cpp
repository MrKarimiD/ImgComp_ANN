#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    blockSize = 8;
    numberOfTrainData = 100;
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::on_train_button_clicked()
{
    //Read Input Image
    QString fileAddress = QFileDialog::getOpenFileName(this,tr("Select Your Source Image"), "/home", tr("Image Files (*.png *.jpg *.bmp)"));
    ui->imgAdd_lineEdit->setText(fileAddress);
    Mat inputFrame, sourceFrame = imread(fileAddress.toStdString());
    cvtColor(sourceFrame, inputFrame, CV_RGB2GRAY);

    // Generating train data set
    srand (time(NULL));
    Mat train_set;
    for(int i = 0; i < numberOfTrainData; i++)
    {
        int cols = rand() % (inputFrame.cols - blockSize);
        int rows = rand() % (inputFrame.rows - blockSize);
        Mat dataItem, tmp = inputFrame(Rect(cols ,rows , blockSize, blockSize));
        tmp.convertTo(dataItem, CV_32F);
        train_set.push_back(dataItem);
    }

    FileStorage fs("nn.yml", FileStorage::WRITE);

    int input_neurons = 64;
    int hidden_neurons = 16;
    int output_neurons = 64;

    Ptr<ml::TrainData> train_data = ml::TrainData::create(train_set, ml::ROW_SAMPLE, train_set);

    Ptr<ml::ANN_MLP> neural_network = ml::ANN_MLP::create();
    neural_network->setTrainMethod(ml::ANN_MLP::BACKPROP);
    neural_network->setBackpropMomentumScale(0.1);
    neural_network->setBackpropWeightScale(0.05);
    neural_network->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, (int)10000, 1e-6));

    Mat layers = Mat(3, 1, CV_32SC1);
    layers.row(0) = Scalar(input_neurons);
    layers.row(1) = Scalar(hidden_neurons);
    layers.row(2) = Scalar(output_neurons);

    neural_network->setLayerSizes(layers);
    neural_network->setActivationFunction(ml::ANN_MLP::SIGMOID_SYM, 1, 1);

    neural_network->train(train_data);
    if (neural_network->isTrained()) {
        neural_network->write(fs);
        qDebug()<< "It's OK!";
    }
}
