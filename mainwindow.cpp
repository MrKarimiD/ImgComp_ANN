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
        Mat tmp, tmp2 = inputFrame(Rect( cols, rows, blockSize, blockSize));
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

    int numOfBlocksInRow = inputFrame.rows / blockSize;
    int numOfBlocksInCol = inputFrame.cols / blockSize;

    //Compressing
    Mat newData, padFrame, compresedData;
    copyMakeBorder( inputFrame, padFrame, 0, blockSize, 0, blockSize, BORDER_REPLICATE);
    for(int r = 0; r*blockSize < inputFrame.rows; r++)
    {
        for(int c = 0; c*blockSize < inputFrame.cols; c++)
        {
            Mat dataItem(blockSize*blockSize, 1, CV_32F);
            Mat tmp, tmp2 = padFrame(Rect(c*blockSize, r*blockSize, blockSize, blockSize));
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
    qDebug()<<"Source frame convereted: Completed!";

    Mat weights = neural_network->getWeights(1);
    for(int k = 0; k < newData.cols; k++ )
    {
        Mat dataItem(hidden_neurons, 1, CV_32F);
        for(int i = 0; i < hidden_neurons; i++)
        {
            double net = double(weights.at<uchar>(0,i));
            for(int j = 0; j < input_neurons ; j++)
            {
                net += weights.at<uchar>(j+1,i)*newData.at<uchar>(j,k);
            }
            double o = sigmoidFunction(net);
            dataItem.at<uchar>(i,0) = o;
        }

        if( compresedData.empty() )
            dataItem.copyTo(compresedData);
        else
            hconcat(compresedData, dataItem, compresedData);
    }
    qDebug()<<"Frame Compressed: Completed!";
    qDebug()<<"Compressed size: "<<compresedData.rows<<" , "<<compresedData.cols;

    Mat outputData, decompresedData;
    //Decompressing
    Mat weights2 = neural_network->getWeights(2);
    for(int k = 0; k < compresedData.cols; k++ )
    {
        Mat dataItem(output_neurons, 1, CV_32F);
        for(int i = 0; i < output_neurons; i++)
        {
            double net = double(weights2.at<uchar>(0,i));
            for(int j = 0; j < hidden_neurons ; j++)
            {
                net += weights2.at<uchar>(j+1,i)*compresedData.at<uchar>(j,k);
            }
            double o = sigmoidFunction(net);
            dataItem.at<uchar>(i,0) = o;
        }

        if( outputData.empty() )
            dataItem.copyTo(outputData);
        else
            hconcat(outputData, dataItem, outputData);
    }
    qDebug()<<"Output Calculated: Completed!";
    qDebug()<<"outputData size: "<<outputData.rows<<" , "<<outputData.cols;

    Mat rowOfData;
    for(int j = 0; j < outputData.cols; j++)
    {
        Mat dataItem;
        for(int i = 0; i*blockSize < blockSize*blockSize; i++)
        {
            Mat eachCol;
            for(int n = 0; n< blockSize;n++)
                eachCol.push_back(outputData.at<uchar>((i*blockSize)+n, j));

            if( dataItem.empty() )
                eachCol.copyTo(dataItem);
            else
                hconcat(dataItem, eachCol, dataItem);
        }

        if( rowOfData.empty() )
            dataItem.copyTo(rowOfData);
        else
            hconcat(rowOfData, dataItem, rowOfData);

        if( rowOfData.cols >= inputFrame.cols )
        {
            if( decompresedData.empty() )
                rowOfData.copyTo(decompresedData);
            else
                vconcat(decompresedData, rowOfData, decompresedData);

            rowOfData.release();
            rowOfData = Mat();
        }
    }
    qDebug()<<"Frame reconstructed: Completed!";
    qDebug()<<"decompresedData size: "<<decompresedData.rows<<" , "<<decompresedData.cols;
}

double MainWindow::sigmoidFunction(double input)
{
    return 1/(1+exp(-input));
}
