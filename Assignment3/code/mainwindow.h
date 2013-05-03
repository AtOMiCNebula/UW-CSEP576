#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QLabel>

template <class T> const T& max ( const T& a, const T& b ) {
  return (b<a)?a:b;     // or: return comp(b,a)?a:b; for the comp version
}

template <class T> const T& min ( const T& a, const T& b ) {
  return (b>a)?a:b;     // or: return comp(b,a)?a:b; for the comp version
}

class MainWindow;

namespace Ui {
    class MainWindow;
}

class CMouseLabel : public QLabel
{
    Q_OBJECT

public:
    MainWindow *window;

protected:
    void mouseMoveEvent(QMouseEvent *ev);
};

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();
    void DrawLine(int r);


private:
    Ui::MainWindow *ui;

    QImage m_InImage1;
    QImage m_InImage2;
    QImage m_DisplayImage1;
    QImage m_DisplayImage2;
    QImage m_GTDisplayImage;
    QImage m_DisparityImage;
    QImage m_SegmentImage;
    QImage m_ErrorImage;
    QImage m_RenderImage;
    QImage m_MatchCostDisplay;

    CMouseLabel m_Image1Display;
    CMouseLabel m_Image2Display;
    CMouseLabel m_GTDisplay;
    CMouseLabel m_ErrorDisplay;
    CMouseLabel m_DisparityDisplay;
    CMouseLabel m_RenderDisplay;
    CMouseLabel m_SegmentDisplay;


    int m_LastRow;
    int m_DisparityScale;
    int m_NumDisparities;
    int m_MinDisparity;
    int m_MaxDisparity;
    int m_SegmentIteration;

    double *m_MatchCost;
    double *m_Disparities;
    double m_DisparityError;

    void DrawDisplayImage();

    // These are functions you need to complete
    void SSD(QImage image1, QImage image2, int minDisparity, int maxDisparity, double *matchCost);
    void SAD(QImage image1, QImage image2, int minDisparity, int maxDisparity, double *matchCost);
    void NCC(QImage image1, QImage image2, int minDisparity, int maxDisparity, int radius, double *matchCost);
    void GaussianBlurMatchScore(double *matchCost, int w, int h, int numDisparities, double sigma);
    void BilateralBlurMatchScore(double *matchCost, int w, int h, int numDisparities,
                                 double sigmaS, double sigmaI, QImage colorImage);
    void SeparableGaussianBlurImage(double *image, int w, int h, double sigma);
    void FindBestDisparity(double *matchCost, double *disparities, int width, int height,
                           int minDisparity, int numDisparities);
    void ComputeSegmentMeans(QImage image, int *segment, int numSegments,
                             double (*meanSpatial)[2], double (*meanColor)[3]);
    void AssignPixelsToSegments(QImage image, int *segment, int numSegments,
                                double (*meanSpatial)[2], double (*meanColor)[3],
                                double spatialSigma, double colorSigma);
    void SegmentAverageMatchCost(int *segment, int numSegments, int width, int height,
                                 int numDisparities, double *matchCost);
    void MagicStereo(QImage image1, QImage image2, int minDisparity, int maxDisparity, double param1, double param2, double *matchCost);

    // These are done for you
    void Segment(QImage image, int gridSize, int numIterations, double spatialSigma, double colorSigma,
                 double *matchCost, int numDisparities, QImage *segmentImage);
    void GridSegmentation(int *segment, int &numSegments, int gridSize, int width, int height);
    void DrawSegments(QImage *segmentImage, int *segment, double (*meanColor)[3]);
    void DisplayDisparities(double *disparities, int disparityScale, int minDisparity,
                            QImage *disparityImage, QImage *errorImage, QImage GTImage, double *m_DisparityError);
    void Render(QImage image, double *disparities, double disparityScale, QImage *renderImage);
    void FillHoles(double *projDisparity, double *projDisparityCt, int width, int height);

private slots:
    void Open();
    void SaveImage();
    void SSD();
    void SAD();
    void NCC();
    void GTOnOff();
    void Gaussian();
    void FindBestDisparity();
    void Bilateral();
    void Render();
    void RenderSlider(int val);
    void Segment();
    void MagicStereo();
};

#endif // MAINWINDOW_H
