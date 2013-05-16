#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QtGui>
#include <QFileDialog>

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    connect(ui->openDatasetButton, SIGNAL(clicked()), this, SLOT(OpenDataSet()));
    connect(ui->openImageButton, SIGNAL(clicked()), this, SLOT(OpenImage()));
    connect(ui->saveImageButton, SIGNAL(clicked()), this, SLOT(SaveImage()));
    connect(ui->averageFaceButton, SIGNAL(clicked()), this, SLOT(AverageFace()));
    connect(ui->integralImageButton, SIGNAL(clicked()), this, SLOT(DisplayIntegralImage()));
    connect(ui->computeFeaturesButton, SIGNAL(clicked()), this, SLOT(ComputeFeatures()));
    connect(ui->adaboostButton, SIGNAL(clicked()), this, SLOT(AdaBoost()));
    connect(ui->saveClassifierButton, SIGNAL(clicked()), this, SLOT(SaveClassifier()));
    connect(ui->openClassifierButton, SIGNAL(clicked()), this, SLOT(OpenClassifier()));
    connect(ui->findFacesButton, SIGNAL(clicked()), this, SLOT(FindFaces()));
    connect(ui->NMSButton, SIGNAL(clicked()), this, SLOT(NMS()));

    ui->faceThresholdSpinBox->setValue(5.0);
    ui->minScaleSpinBox->setValue(30.0);
    ui->maxScaleSpinBox->setValue(50.0);
    ui->xyNMSSpinBox->setValue(30.0);
    ui->scaleNMSSpinBox->setValue(15.0);

    m_DisplayImage = QImage(ui->displayLabel->width(), ui->displayLabel->height(), QImage::Format_RGB32);
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::DrawDisplayImage()
{
    ui->displayLabel->setPixmap(QPixmap::fromImage(m_DisplayImage));
}

void MainWindow::OpenDataSet()
{
    QString fileName = QFileDialog::getOpenFileName(this);
    m_DisplayImage = QImage(ui->displayLabel->width(), ui->displayLabel->height(), QImage::Format_RGB32);

    if (!fileName.isEmpty())
    {
        QFile file(fileName);
        if (!file.open(QIODevice::ReadOnly | QIODevice::Text))
            return;

        QTextStream in(&file);
        QString posName;
        QString negName;
        QString directoryName = fileName;

        directoryName.remove(directoryName.lastIndexOf('/') + 1, directoryName.length());

        QString readText;
        in >> readText;
        m_NumTrainingExamples = readText.toInt();
        in >> readText;
        m_PatchSize = readText.toInt();
        in >> readText;
        m_NumCandidateWeakClassifiers = readText.toInt();
        in >> readText;
        m_NumWeakClassifiers = readText.toInt();

        in >> posName;
        in >> negName;

        QDir faceDirectory(directoryName + posName);
        QDir backgroundDirectory(directoryName + negName);

        m_TrainingData = new double [m_NumTrainingExamples*m_PatchSize*m_PatchSize];
        m_TrainingLabel = new int [m_NumTrainingExamples];

        OpenDataSet(faceDirectory, backgroundDirectory, m_TrainingData, m_TrainingLabel, m_NumTrainingExamples, m_PatchSize);
        DisplayTrainingDataset(&m_DisplayImage, m_TrainingData, m_TrainingLabel, m_NumTrainingExamples, m_PatchSize);
        DrawDisplayImage();
    }
}

void MainWindow::OpenImage()
{
    QString fileName = QFileDialog::getOpenFileName(this);

    if (!fileName.isEmpty())
        m_InImage.load(fileName);

    m_DisplayImage = m_InImage.copy();

    DrawDisplayImage();

}

void MainWindow::SaveImage()
{
    QString fileName = QFileDialog::getSaveFileName(this);

    if (!fileName.isEmpty())
        m_DisplayImage.save(fileName);
}

void MainWindow::SaveClassifier()
{
    QString fileName = QFileDialog::getSaveFileName(this);

    if (!fileName.isEmpty())
        SaveClassifier(fileName);
}

void MainWindow::OpenClassifier()
{
    QString fileName = QFileDialog::getOpenFileName(this);

    m_DisplayImage = QImage(ui->displayLabel->width(), ui->displayLabel->height(), QImage::Format_RGB32);

    if (!fileName.isEmpty())
        OpenClassifier(fileName);

    DisplayClassifiers(&m_DisplayImage, m_WeakClassifiers, m_NumWeakClassifiers);
    DrawDisplayImage();
}

void MainWindow::AverageFace()
{
    DisplayAverageFace(&m_DisplayImage, m_TrainingData, m_TrainingLabel, m_NumTrainingExamples, m_PatchSize);
    DrawDisplayImage();
}

void MainWindow::DisplayIntegralImage()
{
    int w = m_InImage.width();
    int h = m_InImage.height();
    double *integralImage = new double [w*h];
    double *image = new double [w*h];

    ConvertColorToDouble(m_InImage, image, w, h);
    IntegralImage(image, integralImage, w, h);

    DisplayIntegralImage(&m_DisplayImage, integralImage, w, h);
    DrawDisplayImage();

    delete [] integralImage;
}

void MainWindow::ComputeFeatures()
{
    m_CandidateWeakClassifiers = new CWeakClassifiers [m_NumCandidateWeakClassifiers];
    m_Features = new double [m_NumTrainingExamples*m_NumCandidateWeakClassifiers];

    InitializeFeatures(m_CandidateWeakClassifiers, m_NumCandidateWeakClassifiers);
    ComputeTrainingSetFeatures(m_TrainingData, m_Features, m_NumTrainingExamples, m_PatchSize, m_CandidateWeakClassifiers, m_NumCandidateWeakClassifiers);

    DisplayFeatures(&m_DisplayImage, m_Features, m_TrainingLabel, m_NumCandidateWeakClassifiers, m_NumTrainingExamples);
    DrawDisplayImage();
}

void MainWindow::AdaBoost()
{
    m_WeakClassifiers = new CWeakClassifiers[m_NumWeakClassifiers];

    AdaBoost(m_Features, m_TrainingLabel, m_NumTrainingExamples,
                  m_CandidateWeakClassifiers, m_NumCandidateWeakClassifiers, m_WeakClassifiers, m_NumWeakClassifiers);

    DisplayClassifiers(&m_DisplayImage, m_WeakClassifiers, m_NumWeakClassifiers);
    DrawDisplayImage();
}

void MainWindow::FindFaces()
{
    double threshold = ui->faceThresholdSpinBox->value();
    double minScale = ui->minScaleSpinBox->value();
    double maxScale = ui->maxScaleSpinBox->value();

    m_DisplayImage = m_InImage.copy();
    m_FaceDetections.clear();
    FindFaces(m_InImage, m_WeakClassifiers, m_NumWeakClassifiers, threshold, minScale, maxScale, &m_FaceDetections, &m_DisplayImage);
    DrawDisplayImage();
}

void MainWindow::NMS()
{
    double xyThreshold = ui->xyNMSSpinBox->value();
    double scaleThreshold = ui->scaleNMSSpinBox->value();

    m_DisplayImage = m_InImage.copy();
    NMS(&m_FaceDetections, xyThreshold, scaleThreshold, &m_DisplayImage);
    DrawDisplayImage();

}
