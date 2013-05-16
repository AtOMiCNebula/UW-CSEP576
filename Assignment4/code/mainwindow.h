#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QtGui>

//#define NUM_TRAINING_EXAMPLES           1000
//#define PATCH_SIZE                      64
//#define NUM_CANDIDATE_WEAK_CLASSIFIERS  2000
//#define NUM_WEAK_CLASSIFIERS            200

template <class T> const T& max ( const T& a, const T& b ) {
  return (b<a)?a:b;     // or: return comp(b,a)?a:b; for the comp version
}

template <class T> const T& min ( const T& a, const T& b ) {
  return (b>a)?a:b;     // or: return comp(b,a)?a:b; for the comp version
}


namespace Ui {
    class MainWindow;
}

class CWeakClassifiers
{
public:
    double (*m_Box)[2][2];
    double *m_BoxSign;
    int m_NumBoxes;

    double m_Area;
    double m_Threshold;
    double m_Polarity;
    double m_Weight;

    void copy(CWeakClassifiers *wc)
    {
        wc->m_NumBoxes = m_NumBoxes;
        wc->m_Box = new double [m_NumBoxes][2][2];
        wc->m_BoxSign = new double [m_NumBoxes];

        for(int i=0;i<m_NumBoxes;i++)
        {
            wc->m_Box[i][0][0] = m_Box[i][0][0];
            wc->m_Box[i][0][1] = m_Box[i][0][1];
            wc->m_Box[i][1][0] = m_Box[i][1][0];
            wc->m_Box[i][1][1] = m_Box[i][1][1];
            wc->m_BoxSign[i] = m_BoxSign[i];
        }

        wc->m_Area = m_Area;
        wc->m_Threshold = m_Threshold;
        wc->m_Polarity = m_Polarity;
        wc->m_Weight = m_Weight;
    }

};

class CDetection
{
public:
    double m_Score;
    double m_X;
    double m_Y;
    double m_Scale;
};

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

private:
    Ui::MainWindow *ui;

    QImage m_DisplayImage;
    QImage m_InImage;
    double *m_TrainingData;
    int *m_TrainingLabel;
    double *m_Features;
    int m_NumTrainingExamples;
    int m_PatchSize;
    CWeakClassifiers *m_CandidateWeakClassifiers;
    int m_NumCandidateWeakClassifiers;
    CWeakClassifiers *m_WeakClassifiers;
    int m_NumWeakClassifiers;
    QMap<double, CDetection> m_FaceDetections;

    // These are done for you
    void DrawDisplayImage();
    void OpenDataSet(QDir posdirectory, QDir negdirectory, double *trainingData, int *trainingLabel, int numTrainingExamples, int patchSize);
    void DisplayTrainingDataset(QImage *displayImage, double *trainingData, int *trainingLabel, int numTrainingExamples, int patchSize);
    void DisplayIntegralImage(QImage *displayImage, double *integralImage, int w, int h);
    void ConvertColorToDouble(QImage image, double *dImage, int w, int h);
    void InitializeFeatures(CWeakClassifiers *weakClassifiers, int numWeakClassifiers);
    void ComputeTrainingSetFeatures(double *trainingData, double *features,
                                    int numTrainingExamples, int patchSize, CWeakClassifiers *weakClassifiers, int numWeakClassifiers);
    void DisplayFeatures(QImage *displayImage, double *features, int *trainingLabel, int numFeatures, int numTrainingExamples);
    void AdaBoost(double *features, int *trainingLabel, int numTrainingExamples,
                  CWeakClassifiers *candidateWeakClassifiers, int numCandidateWeakClassifiers, CWeakClassifiers *weakClassifiers, int numWeakClassifiers);
    void SaveClassifier(QString fileName);
    void OpenClassifier(QString fileName);
    void DisplayClassifiers(QImage *displayImage, CWeakClassifiers *weakClassifiers, int numWeakClassifiers);
    void FindFaces(QImage inImage, CWeakClassifiers *weakClassifiers, int numWeakClassifiers, double threshold,
                   double minScale, double maxScale, QMap<double, CDetection> *faceDetections, QImage *displayImage);
    void DrawFace(QImage *displayImage, QMap<double, CDetection> *faceDetections);


    // These are the functions you need to complete
    // Step 1
    void DisplayAverageFace(QImage *displayImage, double *trainingData, int *trainingLabel, int numTrainingExamples, int patchSize);
    void IntegralImage(double *image, double *integralImage, int w, int h);
    void ComputeFeatures(double *integralImage, int c0, int r0, int size, double *features, CWeakClassifiers *weakClassifiers, int numWeakClassifiers, int w);
    double SumBox(double *integralImage, double x0, double y0, double x1, double y1, int w);
    double BilinearInterpolation(double *image, double x, double y, int w);

    // Step 2
    double FindBestClassifier(int *featureSortIdx, double *features, int *trainingLabel, double *dataWeights,
                              int numTrainingExamples, CWeakClassifiers candidateWeakClassifiers, CWeakClassifiers *bestClassifier);
    void UpdateDataWeights(double *features, int *trainingLabel, CWeakClassifiers weakClassifier, double *dataWeights, int numTrainingExamples);

    // Step 3
    double ClassifyBox(double *integralImage, int c0, int r0, int size, CWeakClassifiers *weakClassifiers, int numWeakClassifiers, int w);
    void NMS(QMap<double, CDetection> *faceDetections, double xyThreshold, double scaleThreshold, QImage *displayImage);


private slots:
    void OpenDataSet();
    void OpenImage();
    void SaveImage();
    void AverageFace();
    void DisplayIntegralImage();
    void ComputeFeatures();
    void AdaBoost();
    void SaveClassifier();
    void OpenClassifier();
    void FindFaces();
    void NMS();

};

#endif // MAINWINDOW_H
