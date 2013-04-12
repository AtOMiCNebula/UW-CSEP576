#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>

template <class T> const T& max ( const T& a, const T& b ) {
  return (b<a)?a:b;     // or: return comp(b,a)?a:b; for the comp version
}

template <class T> const T& min ( const T& a, const T& b ) {
  return (b>a)?a:b;     // or: return comp(b,a)?a:b; for the comp version
}

namespace Ui {
    class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

private:
    Ui::MainWindow *ui;
    QImage inImage;
    QImage outImage;
    QImage displayImage;
    double xScroll;
    double yScroll;
    bool zoomChanged;

    void DrawDisplayImage();
    void BlackWhiteImage(QImage *image);
    void AddNoise(QImage *image, double mag, bool colorNoise);
    void MeanBlurImage(QImage *image, int radius);
    void GaussianBlurImage(QImage *image, double sigma);
    void SeparableGaussianBlurImage(QImage *image, double sigma);
    void FirstDerivImage(QImage *image, double sigma);
    void SecondDerivImage(QImage *image, double sigma);
    void SharpenImage(QImage *image, double sigma, double alpha);
    void SobelImage(QImage *image);
    void HoughImage(QImage *image);
    void CrazyImage(QImage *image);
    void HalfImage(QImage &image);
    void FindPeaksImage(QImage *image, double thres);
    void BilateralImage(QImage *image, double sigmaS, double sigmaI);
    void MedianImage(QImage *image, int radius);
    void RotateImage(QImage *image, double orien);

    void BilinearInterpolation(QImage *image, double x, double y, double rgb[3]);


private slots:
    void OpenImage();
    void SaveImage();
    void SaveDisplayImage();
    void ResetImage();
    void ToggleImage();
    void AddNoise();
    void BlackWhiteImage();
    void MeanBlurImage();
    void GaussianBlurImage();
    void FirstDerivImage();
    void SecondDerivImage();
    void SharpenImage();
    void SobelImage();
    void HalfImage();
    void RotateImage();
    void Brightness(int val);
    void Zoom(int val);
    void Scroll(int val);
    void BilateralImage();
    void FindPeaksImage();
    void MedianImage();
    void HoughImage();
    void CrazyImage();

};

#endif // MAINWINDOW_H
