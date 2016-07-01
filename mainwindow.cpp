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
    qDebug()<<"Read Image: Complete!";

    // Generating train data set
    srand (time(NULL));
    Mat train_set;
    for(int i = 0; i < numberOfTrainData; i++)
    {
        int cols = rand() % (inputFrame.cols - blockSize);
        int rows = rand() % (inputFrame.rows - blockSize);
        Mat tmp, tmp2 = inputFrame(Rect(rows ,cols , blockSize, blockSize));
        Mat dataItem(blockSize*blockSize, 1, CV_32F);
        tmp2.convertTo(tmp, CV_32F);
        for(int r = 0; r < blockSize; r++)
        {
            for(int c = 0; c < blockSize; c++)
            {
                dataItem.at<uchar>(r*blockSize + c ,0) = tmp.at<uchar>(r ,c);
            }
        }

        if( train_set.empty() )
            dataItem.copyTo(train_set);
        else
            hconcat(train_set, dataItem, train_set);
    }
    qDebug()<<"Generating train data: Complete!";

    //Training
    FileStorage fs("nn.yml", FileStorage::WRITE);

    int input_neurons = 64; // It should be equals to number of cols
    int hidden_neurons = 16;
    int output_neurons = 64;

    Ptr<ml::TrainData> train_data = ml::TrainData::create(train_set, ml::COL_SAMPLE, train_set);

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
    fs.release();
    qDebug()<<"Training: Complete!";

    //Compressing
    Mat newData, padFrame, compresedData;
    copyMakeBorder( inputFrame, padFrame, 0, blockSize, 0, blockSize, BORDER_REPLICATE);
    for(int r = 0; r*blockSize < inputFrame.rows; r++)
    {
        for(int c = 0; c*blockSize < inputFrame.cols; c++)
        {
            Mat dataItem(blockSize*blockSize, 1, CV_32F);
            Mat tmp, tmp2 = padFrame(Rect(r*blockSize ,c*blockSize , blockSize, blockSize));
            tmp2.convertTo(tmp, CV_32F);
            for(int i = 0; i < blockSize; i++)
            {
                for(int j = 0; j < blockSize; j++)
                {
                    dataItem.at<uchar>(i*blockSize + j ,0) = tmp.at<uchar>(i ,j);
                }
            }

            if( newData.empty() )
                dataItem.copyTo(newData);
            else
                hconcat(newData, dataItem, newData);
        }
    }
}
