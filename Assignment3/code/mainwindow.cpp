#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QtGui>
#include <QFileDialog>

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    connect(ui->openButton, SIGNAL(clicked()), this, SLOT(Open()));
    connect(ui->saveButton, SIGNAL(clicked()), this, SLOT(SaveImage()));
    connect(ui->SSDButton, SIGNAL(clicked()), this, SLOT(SSD()));
    connect(ui->SADButton, SIGNAL(clicked()), this, SLOT(SAD()));
    connect(ui->NCCButton, SIGNAL(clicked()), this, SLOT(NCC()));
    connect(ui->GTCheckBox, SIGNAL(clicked()), this, SLOT(GTOnOff()));
    connect(ui->gaussianButton, SIGNAL(clicked()), this, SLOT(Gaussian()));
    connect(ui->maxButton, SIGNAL(clicked()), this, SLOT(FindBestDisparity()));
    connect(ui->bilateralButton, SIGNAL(clicked()), this, SLOT(Bilateral()));
    connect(ui->segmentButton, SIGNAL(clicked()), this, SLOT(Segment()));
    connect(ui->renderButton, SIGNAL(clicked()), this, SLOT(Render()));
    connect(ui->renderSlider, SIGNAL(valueChanged(int)), this, SLOT(RenderSlider(int)));
    connect(ui->magicButton, SIGNAL(clicked()), this, SLOT(MagicStereo()));

    ui->GTCheckBox->setChecked(true);
    ui->pixelErrorLabel->setText("");
    ui->gaussianSigmaSpinBox->setValue(1.0);
    ui->biSigmaSSpinBox->setValue(1.0);
    ui->biSigmaISpinBox->setValue(20.0);
    ui->renderSlider->setValue(100);
    ui->NCCRadiusBox->setValue(2);
    ui->segmentGridBox->setValue(20);
    ui->segmentColorSpinBox->setValue(20.0);
    ui->segmentSpatialSpinBox->setValue(6.0);
    ui->segmentIterBox->setValue(4);

    m_Image1Display.setParent(ui->tab);
    m_Image2Display.setParent(ui->tab_2);
    m_GTDisplay.setParent(ui->tab_4);
    m_DisparityDisplay.setParent(ui->tab_3);
    m_ErrorDisplay.setParent(ui->tab_5);
    m_RenderDisplay.setParent(ui->tab_6);
    m_SegmentDisplay.setParent(ui->tab_7);

    m_Image1Display.window = this;
    m_Image2Display.window = this;
    m_GTDisplay.window = this;
    m_DisparityDisplay.window = this;
    m_ErrorDisplay.window = this;
    m_RenderDisplay.window = this;
    m_SegmentDisplay.window = this;

    ui->tabWidget->setCurrentIndex(0);

    m_LastRow = 0;
    m_SegmentIteration = 0;
    m_MatchCost = NULL;



}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::DrawDisplayImage()
{
    m_Image1Display.setPixmap(QPixmap::fromImage(m_DisplayImage1));
    m_Image2Display.setPixmap(QPixmap::fromImage(m_DisplayImage2));
    m_GTDisplay.setPixmap(QPixmap::fromImage(m_GTDisplayImage));
    m_DisparityDisplay.setPixmap(QPixmap::fromImage(m_DisparityImage));
    m_ErrorDisplay.setPixmap(QPixmap::fromImage(m_ErrorImage));
    m_RenderDisplay.setPixmap(QPixmap::fromImage(m_RenderImage));
    m_SegmentDisplay.setPixmap(QPixmap::fromImage(m_SegmentImage));

}



void MainWindow::Open()
{
    const QString title;
    int minDisparity = 0;
    int maxDisparity = 0;

    QString fileName = QFileDialog::getOpenFileName(this, title);

    if (!fileName.isEmpty())
    {
        QFile file(fileName);
        if (!file.open(QIODevice::ReadOnly | QIODevice::Text))
            return;

        QTextStream in(&file);
        QString imageName;
        QString directoryName = fileName;

        directoryName.remove(directoryName.lastIndexOf('/') + 1, directoryName.length());


        in >> imageName;
        m_InImage1.load(directoryName + imageName);
        m_DisplayImage1 = m_InImage1.copy();

        in >> imageName;
        m_InImage2.load(directoryName + imageName);
        m_DisplayImage2 = m_InImage2.copy();

        in >> imageName;
        if(!imageName.isEmpty())
        {
            m_GTDisplayImage.load(directoryName + imageName);
        }

        in >> minDisparity;
        in >> maxDisparity;
        in >> m_DisparityScale;

        DrawDisplayImage();

    }

    m_MinDisparity = minDisparity;
    m_MaxDisparity = maxDisparity;

    char text[100];
    sprintf(text, "Min Disparity: %d", m_MinDisparity);
    QString textmin(text);
    ui->minDisparitylabel->setText(textmin);

    sprintf(text, "Max Disparity: %d", m_MaxDisparity);
    QString textmax(text);
    ui->maxDisparityLabel->setText(textmax);

    m_NumDisparities = maxDisparity - minDisparity;
    int width = m_InImage1.width();
    int height = m_InImage1.height();

    if(m_NumDisparities > 0)
    {
        if(m_MatchCost != NULL)
        {
            delete [] m_MatchCost;
            delete [] m_Disparities;
        }

        m_MatchCost = new double [m_NumDisparities*height*width];
        m_Disparities = new double [height*width];
        m_MatchCostDisplay = QImage(width, m_NumDisparities, QImage::Format_RGB32);
        m_DisparityImage = QImage(width, height, QImage::Format_RGB32);
        m_ErrorImage = QImage(width, height, QImage::Format_RGB32);
        m_RenderImage = QImage(width, height, QImage::Format_RGB32);
        m_SegmentImage = QImage(width, height, QImage::Format_RGB32);

        m_MatchCostDisplay.fill(qRgb(0,0,0));
        m_DisparityImage.fill(qRgb(0,0,0));
        m_ErrorImage.fill(qRgb(0,0,0));
        m_RenderImage.fill(qRgb(0,0,0));
        m_SegmentImage.fill(qRgb(0,0,0));

        m_Image1Display.setFixedSize(width, height);
        m_Image2Display.setFixedSize(width, height);
        m_GTDisplay.setFixedSize(width, height);
        m_DisparityDisplay.setFixedSize(width, height);
        m_ErrorDisplay.setFixedSize(width, height);
        m_RenderDisplay.setFixedSize(width, height);
        m_SegmentDisplay.setFixedSize(width, height);
    }

   DrawDisplayImage();
   ui->tabWidget->setCurrentIndex(0);
}

void MainWindow::SaveImage()
{
    const QString title;

    QString fileName = QFileDialog::getSaveFileName(this, title);

    if(ui->tabWidget->currentIndex() ==  0)
    {
        if (!fileName.isEmpty())
            m_DisplayImage1.save(fileName);
    }

    if(ui->tabWidget->currentIndex() ==  1)
    {
        if (!fileName.isEmpty())
            m_DisplayImage2.save(fileName);
    }

    if(ui->tabWidget->currentIndex() ==  2)
    {
        if (!fileName.isEmpty())
            m_SegmentImage.save(fileName);
    }

    if(ui->tabWidget->currentIndex() ==  3)
    {
        if (!fileName.isEmpty())
            m_GTDisplayImage.save(fileName);
    }

    if(ui->tabWidget->currentIndex() ==  4)
    {
        if (!fileName.isEmpty())
            m_DisparityImage.save(fileName);
    }

    if(ui->tabWidget->currentIndex() ==  5)
    {
        if (!fileName.isEmpty())
            m_ErrorImage.save(fileName);
    }

    if(ui->tabWidget->currentIndex() ==  6)
    {
        if (!fileName.isEmpty())
            m_RenderImage.save(fileName);
    }
}

void MainWindow::SSD()
{
    SSD(m_InImage1, m_InImage2, m_MinDisparity, m_MaxDisparity, m_MatchCost);
}

void MainWindow::SAD()
{
    SAD(m_InImage1, m_InImage2, m_MinDisparity, m_MaxDisparity, m_MatchCost);
}

void MainWindow::NCC()
{
    int radius = ui->NCCRadiusBox->value();

    NCC(m_InImage1, m_InImage2, m_MinDisparity, m_MaxDisparity, radius, m_MatchCost);
}

void MainWindow::Gaussian()
{
    double sigma = ui->gaussianSigmaSpinBox->value();

    GaussianBlurMatchScore(m_MatchCost, m_InImage1.width(), m_InImage1.height(),
                           m_NumDisparities, sigma);

}

void MainWindow::Bilateral()
{
    double sigmaS = ui->biSigmaSSpinBox->value();
    double sigmaI = ui->biSigmaISpinBox->value();

    BilateralBlurMatchScore(m_MatchCost, m_InImage1.width(), m_InImage1.height(),
                           m_NumDisparities, sigmaS, sigmaI, m_InImage1);

}

void CMouseLabel::mouseMoveEvent ( QMouseEvent * ev )
{
    window->DrawLine(ev->y());

    QWidget::mouseMoveEvent(ev);
}

void MainWindow::GTOnOff()
{
    DrawLine(m_LastRow);
}

void MainWindow::DrawLine(int r)
{
    int c, d;
    QImage image;
    QRgb pixel;

    r = min(m_InImage1.height()-1, max(0, r));
    m_LastRow = r;

    if(ui->tabWidget->currentIndex() ==  0)
    {
        image = m_DisplayImage1;

        for(c=0;c<image.width();c++)
        {
            image.setPixel(c, r, qRgb(255, 0, 0));
        }

        m_Image1Display.setPixmap(QPixmap::fromImage(image));
    }

    if(ui->tabWidget->currentIndex() ==  1)
    {
        image = m_DisplayImage2;

        for(c=0;c<image.width();c++)
        {
            image.setPixel(c, r, qRgb(255, 0, 0));
        }

        m_Image2Display.setPixmap(QPixmap::fromImage(image));
    }

    if(ui->tabWidget->currentIndex() ==  2)
    {
        image = m_SegmentImage;

        for(c=0;c<image.width();c++)
        {
            image.setPixel(c, r, qRgb(255, 0, 0));
        }

        m_SegmentDisplay.setPixmap(QPixmap::fromImage(image));
    }

    if(ui->tabWidget->currentIndex() ==  3)
    {
        image = m_GTDisplayImage;

        for(c=0;c<image.width();c++)
        {
            image.setPixel(c, r, qRgb(255, 0, 0));
        }

        m_GTDisplay.setPixmap(QPixmap::fromImage(image));
    }

    if(ui->tabWidget->currentIndex() ==  4)
    {
        image = m_DisparityImage;

        for(c=0;c<image.width();c++)
        {
            image.setPixel(c, r, qRgb(255, 0, 0));
        }

        m_DisparityDisplay.setPixmap(QPixmap::fromImage(image));
    }

    if(ui->tabWidget->currentIndex() ==  5)
    {
        image = m_ErrorImage;

        for(c=0;c<image.width();c++)
        {
            image.setPixel(c, r, qRgb(255, 0, 0));
        }

        m_ErrorDisplay.setPixmap(QPixmap::fromImage(image));
    }

    if(ui->tabWidget->currentIndex() ==  6)
    {
        image = m_RenderImage;

        for(c=0;c<image.width();c++)
        {
            image.setPixel(c, r, qRgb(255, 0, 0));
        }

        m_RenderDisplay.setPixmap(QPixmap::fromImage(image));
    }

    int w = m_InImage1.width();
    int h = m_InImage1.height();

    float avg = 0.0000001;
    float avgCt = 0.0000001;

    for(d=0;d<m_NumDisparities;d+=2)
        for(c=0;c<w;c+=2)
        {
            avg += m_MatchCost[d*w*h + r*w + c];
            avgCt++;
        }

    avg /= avgCt;

    for(d=0;d<m_NumDisparities;d++)
        for(c=0;c<w;c++)
        {
            double score = m_MatchCost[d*w*h + r*w + c];

            score /= avg;
            score *= 100.0;
            score = min(255.0, max(0.0, score));
            int iscore = (int) score;

            m_MatchCostDisplay.setPixel(c, d, qRgb(iscore, iscore, iscore));
        }

    if(ui->GTCheckBox->isChecked())
    {
        for(c=0;c<w;c++)
        {
            pixel = m_GTDisplayImage.pixel(c, r);
            d = qGreen(pixel)/m_DisparityScale;

            m_MatchCostDisplay.setPixel(c, d, qRgb(255, 0, 0));
        }
    }

    ui->matchCostDisplay->setPixmap(QPixmap::fromImage(m_MatchCostDisplay));

}

void MainWindow::FindBestDisparity()
{
    FindBestDisparity(m_MatchCost, m_Disparities, m_InImage1.width(), m_InImage1.height(), m_MinDisparity, m_NumDisparities);

    DisplayDisparities(m_Disparities, m_DisparityScale, m_MinDisparity, &m_DisparityImage, &m_ErrorImage, m_GTDisplayImage, &m_DisparityError);

    DrawDisplayImage();

    char errorc[100];
    sprintf(errorc, "%0.2lf", 100.0*m_DisparityError);
    QString error(errorc);
    ui->pixelErrorLabel->setText(error);
    ui->tabWidget->setCurrentIndex(4);


}

void MainWindow::Segment()
{
    double cSigma = ui->segmentColorSpinBox->value();
    double sSigma = ui->segmentSpatialSpinBox->value();
    int gridSize = ui->segmentGridBox->value();
    int numIterations = ui->segmentIterBox->value();

    m_SegmentImage = m_InImage1.copy();

    Segment(m_InImage1, gridSize, numIterations, sSigma, cSigma, m_MatchCost, m_NumDisparities, &m_SegmentImage);

    DrawDisplayImage();
    ui->tabWidget->setCurrentIndex(2);

}

void MainWindow::Render()
{
    double disparityScale = (double) ui->renderSlider->value()/100.0;

    Render(m_InImage1, m_Disparities, disparityScale, &m_RenderImage);

    DrawDisplayImage();
    ui->tabWidget->setCurrentIndex(6);

}

void MainWindow::RenderSlider(int val)
{
    double disparityScale = (double) ui->renderSlider->value()/100.0;

    Render(m_InImage1, m_Disparities, disparityScale, &m_RenderImage);

    m_RenderDisplay.setPixmap(QPixmap::fromImage(m_RenderImage));
    ui->tabWidget->setCurrentIndex(6);

}

void MainWindow::MagicStereo()
{
    double param1 = ui->magicParam1SpinBox->value();
    double param2 = ui->magicParam2SpinBox->value();

    MagicStereo(m_InImage1, m_InImage2, m_MinDisparity, m_MaxDisparity, param1, param2, m_MatchCost);

}

