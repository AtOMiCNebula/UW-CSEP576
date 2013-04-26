#include "mainwindow.h"
#include "math.h"
#include "ui_mainwindow.h"
#include <QtGui>


// The first four functions provide example code to help get you started

// Convert an image to grey-scale
void MainWindow::BlackWhiteImage(QImage *image)
{
    int r, c;
    QRgb pixel;

    for(r=0;r<image->height();r++)
    {
        for(c=0;c<image->width();c++)
        {
            pixel = image->pixel(c, r);
            double red = (double) qRed(pixel);
            double green = (double) qGreen(pixel);
            double blue = (double) qBlue(pixel);

            // Compute intensity from colors - these are common weights
            double intensity = 255.0 - (0.3*red + 0.6*green + 0.1*blue);

            image->setPixel(c, r, qRgb( (int) intensity, (int) intensity, (int) intensity));
        }
    }
}

// Add random noise to the image
void MainWindow::AddNoise(QImage *image, double mag, bool colorNoise)
{
    int r, c;
    QRgb pixel;
    int noiseMag = mag;
    noiseMag *= 2;

    for(r=0;r<image->height();r++)
    {
        for(c=0;c<image->width();c++)
        {
            pixel = image->pixel(c, r);
            int red = qRed(pixel);
            int green = qGreen(pixel);
            int blue = qBlue(pixel);

            // If colorNoise, add color independently to each channel
            if(colorNoise)
            {
                red += rand()%noiseMag - noiseMag/2;
                green += rand()%noiseMag - noiseMag/2;
                blue += rand()%noiseMag - noiseMag/2;
            }
            else
            {
                int noise = rand()%noiseMag - noiseMag/2;

                red += noise;
                green += noise;
                blue += noise;
            }

            // Make sure we don't over or under saturate
            red = min(255, max(0, red));
            green = min(255, max(0, green));
            blue = min(255, max(0, blue));

            image->setPixel(c, r, qRgb( red, green, blue));
        }
    }
}

// Here is an example of blurring an image using a mean or box filter with the specified radius.
// This could be implemented using separable filters to make it much more efficient, but it is not.
void MainWindow::MeanBlurImage(QImage *image, int radius)
{
    if(radius == 0)
        return;

    int r, c, rd, cd, i;
    QRgb pixel;
    int size = 2*radius + 1;
    QImage buffer;
    int w = image->width();
    int h = image->height();

    // This creates an image of size (w + 2*radius, h + 2*radius) with black borders.
    // This could be improved by filling the pixels using a different padding technique (reflected, fixed, etc.)
    buffer = image->copy(-radius, -radius, w + 2*radius, h + 2*radius);

    // Compute kernel to convolve with the image.
    double *kernel = new double [size];

    for(i=0;i<size;i++)
    {
        kernel[i] = 1.0;
    }

    // Make sure kernel sums to 1
    double denom = 0.000001;
    for(i=0;i<size;i++)
        denom += kernel[i];
    for(i=0;i<size;i++)
        kernel[i] /= denom;

    // For each pixel in the image...
    for(r=0;r<h;r++)
    {
        for(c=0;c<w;c++)
        {
            double pixRGB[3];

            pixRGB[0] = 0.0;
            pixRGB[1] = 0.0;
            pixRGB[2] = 0.0;

            // Convolve the kernel at each pixel
            for(rd=-radius;rd<=radius;rd++)
                for(cd=-radius;cd<=radius;cd++)
                {
                     pixel = buffer.pixel(c + cd + radius, r + rd + radius);
                     double weight = kernel[rd + radius]*kernel[cd + radius];

                     pixRGB[0] += weight*(double) qRed(pixel);
                     pixRGB[1] += weight*(double) qGreen(pixel);
                     pixRGB[2] += weight*(double) qBlue(pixel);
                }

            // Store mean pixel in the new image.
            image->setPixel(c, r, qRgb((int) round(pixRGB[0]), (int) round(pixRGB[1]), (int) round(pixRGB[2])));
        }
    }

    delete [] kernel;
}

// Downsample the image by 1/2
void MainWindow::HalfImage(QImage &image)
{
    QImage buffer;
    int w = image.width();
    int h = image.height();
    int r, c;

    buffer = image.copy();

    // Reduce the image size.
    image = QImage(w/2, h/2, QImage::Format_RGB32);

    // Copy every other pixel
    for(r=0;r<h/2;r++)
        for(c=0;c<w/2;c++)
        {
             image.setPixel(c, r, buffer.pixel(c*2, r*2));
        }
}



void MainWindow::GaussianBlurImage(QImage *image, double sigma)
{
    int r, c, rd, cd, i;
    QRgb pixel;
    int radius = max(1, (int) (sigma*3.0));
    int size = 2*radius + 1;
    QImage buffer;
    int w = image->width();
    int h = image->height();

    printf("rad %d\n", radius);

    buffer = image->copy();

    if(sigma == 0.0)
        return;

    double *kernel = new double [size];

    for(i=0;i<size;i++)
    {
        double dist = (double) (i - radius);

        kernel[i] = exp(-(dist*dist)/(2.0*sigma*sigma));
    }

    double denom = 0.000001;

    for(i=0;i<size;i++)
        denom += kernel[i];
    for(i=0;i<size;i++)
        kernel[i] /= denom;

    for(r=0;r<h;r++)
    {
        for(c=0;c<w;c++)
        {

            double pixRGB[3];

            pixRGB[0] = 0.5;
            pixRGB[1] = 0.5;
            pixRGB[2] = 0.5;
            double denom = 0.0;

            for(rd=-radius;rd<=radius;rd++)
                for(cd=-radius;cd<=radius;cd++)
                    if(r + rd >= 0 && r + rd < h && c + cd >= 0 && c + cd < w)
                {
                     pixel = buffer.pixel(c+cd, r+rd);
                     double weight = kernel[rd + radius]*kernel[cd + radius];

                     pixRGB[0] += weight*(double) qRed(pixel);
                     pixRGB[1] += weight*(double) qGreen(pixel);
                     pixRGB[2] += weight*(double) qBlue(pixel);
                     denom += weight;
                }

            pixRGB[0] /= denom;
            pixRGB[1] /= denom;
            pixRGB[2] /= denom;

            image->setPixel(c, r, qRgb( (int) pixRGB[0], (int) pixRGB[1], (int) pixRGB[2]));
        }
    }


    delete [] kernel;
}

void MainWindow::SeparableGaussianBlurImage(QImage *image, double sigma)
{
    int r, c, rd, cd, i;
    QRgb pixel;
    int radius = max(1, (int) (sigma*3.0));
    int size = 2*radius + 1;
    QImage buffer;
    int w = image->width();
    int h = image->height();

    printf("rad %d\n", radius);

    buffer = image->copy();

    if(sigma == 0.0)
        return;

    double *kernel = new double [size];

    for(i=0;i<size;i++)
    {
        double dist = (double) (i - radius);

        kernel[i] = exp(-(dist*dist)/(2.0*sigma*sigma));
    }

    double denom = 0.000001;

    for(i=0;i<size;i++)
        denom += kernel[i];
    for(i=0;i<size;i++)
        kernel[i] /= denom;

    for(r=0;r<h;r++)
    {
        for(c=0;c<w;c++)
        {

            double pixRGB[3];

            pixRGB[0] = 0.5;
            pixRGB[1] = 0.5;
            pixRGB[2] = 0.5;
            double denom = 0.0;

            for(rd=-radius;rd<=radius;rd++)
                if(r + rd >= 0 && r + rd < h)
                {
                     pixel = buffer.pixel(c, r + rd);
                     double weight = kernel[rd + radius];

                     pixRGB[0] += weight*(double) qRed(pixel);
                     pixRGB[1] += weight*(double) qGreen(pixel);
                     pixRGB[2] += weight*(double) qBlue(pixel);
                     denom += weight;
                }

            pixRGB[0] /= denom/1.0;
            pixRGB[1] /= denom/1.0;
            pixRGB[2] /= denom/1.0;

            image->setPixel(c, r, qRgb( (int) pixRGB[0], (int) pixRGB[1], (int) pixRGB[2]));
        }
    }

    buffer = image->copy();

    for(r=0;r<h;r++)
    {
        for(c=0;c<w;c++)
        {

            double pixRGB[3];

            pixRGB[0] = 0.0;
            pixRGB[1] = 0.0;
            pixRGB[2] = 0.0;
            double denom = 0.0;

            for(cd=-radius;cd<=radius;cd++)
                if(c + cd >= 0 && c + cd < w)
                {
                     pixel = buffer.pixel(c+cd, r);
                     double weight = kernel[cd + radius];

                     pixRGB[0] += weight*(double) qRed(pixel);
                     pixRGB[1] += weight*(double) qGreen(pixel);
                     pixRGB[2] += weight*(double) qBlue(pixel);
                     denom += weight;
                }

            pixRGB[0] /= denom/1.0;
            pixRGB[1] /= denom/1.0;
            pixRGB[2] /= denom/1.0;

            image->setPixel(c, r, qRgb( (int) pixRGB[0], (int) pixRGB[1], (int) pixRGB[2]));
        }
    }


    delete [] kernel;
}

void MainWindow::FirstDerivImage(QImage *image, double sigma)
{
    QImage buffer;
    int w = image->width();
    int h = image->height();
    int r, c;
    QRgb pixel;


    buffer = image->copy();
    for(r=1;r<h-1;r++)
    {
        for(c=1;c<w-1;c++)
        {
            double pixRGB[3];

            pixRGB[0] = 0.0;
            pixRGB[1] = 0.0;
            pixRGB[2] = 0.0;

            pixel = buffer.pixel(c+1, r);

            pixRGB[0] += (double) qRed(pixel);
            pixRGB[1] += (double) qGreen(pixel);
            pixRGB[2] += (double) qBlue(pixel);

            pixel = buffer.pixel(c-1, r);

            pixRGB[0] -= (double) qRed(pixel);
            pixRGB[1] -= (double) qGreen(pixel);
            pixRGB[2] -= (double) qBlue(pixel);

            pixRGB[0] += 128.0;
            pixRGB[1] += 128.0;
            pixRGB[2] += 128.0;

            pixRGB[0] = min(255.0, max(0.0, pixRGB[0]));
            pixRGB[1] = min(255.0, max(0.0, pixRGB[1]));
            pixRGB[2] = min(255.0, max(0.0, pixRGB[2]));

            image->setPixel(c, r, qRgb( (int) pixRGB[0], (int) pixRGB[1], (int) pixRGB[2]));

        }
    }

    GaussianBlurImage(image, sigma);
}

void MainWindow::SecondDerivImage(QImage *image, double sigma)
{
    QImage buffer;
    int w = image->width();
    int h = image->height();
    int r, c;
    QRgb pixel;


    buffer = image->copy();
    for(r=1;r<h-1;r++)
    {
        for(c=1;c<w-1;c++)
        {
            double pixRGB[3];

            pixRGB[0] = 0.0;
            pixRGB[1] = 0.0;
            pixRGB[2] = 0.0;

            pixel = buffer.pixel(c+1, r);

            pixRGB[0] += (double) qRed(pixel);
            pixRGB[1] += (double) qGreen(pixel);
            pixRGB[2] += (double) qBlue(pixel);

            pixel = buffer.pixel(c, r+1);

            pixRGB[0] += (double) qRed(pixel);
            pixRGB[1] += (double) qGreen(pixel);
            pixRGB[2] += (double) qBlue(pixel);

            pixel = buffer.pixel(c, r);

            pixRGB[0] -= 4.0*(double) qRed(pixel);
            pixRGB[1] -= 4.0*(double) qGreen(pixel);
            pixRGB[2] -= 4.0*(double) qBlue(pixel);

            pixel = buffer.pixel(c-1, r);

            pixRGB[0] += (double) qRed(pixel);
            pixRGB[1] += (double) qGreen(pixel);
            pixRGB[2] += (double) qBlue(pixel);

            pixel = buffer.pixel(c, r-1);

            pixRGB[0] += (double) qRed(pixel);
            pixRGB[1] += (double) qGreen(pixel);
            pixRGB[2] += (double) qBlue(pixel);

            pixRGB[0] += 128.0;
            pixRGB[1] += 128.0;
            pixRGB[2] += 128.0;

            pixRGB[0] = min(255.0, max(0.0, pixRGB[0]));
            pixRGB[1] = min(255.0, max(0.0, pixRGB[1]));
            pixRGB[2] = min(255.0, max(0.0, pixRGB[2]));

            image->setPixel(c, r, qRgb( (int) pixRGB[0], (int) pixRGB[1], (int) pixRGB[2]));

        }
    }

    GaussianBlurImage(image, sigma);
}

void MainWindow::SharpenImage(QImage *image, double sigma, double mag)
{
    QImage buffer;
    int w = image->width();
    int h = image->height();
    int r, c;
    QRgb pixel;


    buffer = image->copy();

    SecondDerivImage(&buffer, sigma);

    for(r=1;r<h-1;r++)
    {
        for(c=1;c<w-1;c++)
        {
            double pixRGB[3];

            pixRGB[0] = 0.0;
            pixRGB[1] = 0.0;
            pixRGB[2] = 0.0;

            pixel = image->pixel(c, r);

            pixRGB[0] += (double) qRed(pixel);
            pixRGB[1] += (double) qGreen(pixel);
            pixRGB[2] += (double) qBlue(pixel);

            pixel = buffer.pixel(c, r);

            pixRGB[0] += mag*((double) qRed(pixel) - 128.0);
            pixRGB[1] += mag*((double) qGreen(pixel) - 128.0);
            pixRGB[2] += mag*((double) qBlue(pixel) - 128.0);

            pixRGB[0] = min(255.0, max(0.0, pixRGB[0]));
            pixRGB[1] = min(255.0, max(0.0, pixRGB[1]));
            pixRGB[2] = min(255.0, max(0.0, pixRGB[2]));

            image->setPixel(c, r, qRgb( (int) pixRGB[0], (int) pixRGB[1], (int) pixRGB[2]));

        }
    }

}

void MainWindow::SobelImage(QImage *image)
{
    QImage buffer;
    int w = image->width();
    int h = image->height();
    int r, c, rd, cd;
    QRgb pixel;

    double filterX[3][3];
    double filterY[3][3];

    filterX[0][0] = -1.0;
    filterX[1][0] = -2.0;
    filterX[2][0] = -1.0;

    filterX[0][1] = 0.0;
    filterX[1][1] = 0.0;
    filterX[2][1] = 0.0;

    filterX[0][2] = 1.0;
    filterX[1][2] = 2.0;
    filterX[2][2] = 1.0;

    filterY[0][0] = -1.0;
    filterY[1][0] = 0.0;
    filterY[2][0] = 1.0;

    filterY[0][1] = -2.0;
    filterY[1][1] = 0.0;
    filterY[2][1] = 2.0;

    filterY[0][2] = -1.0;
    filterY[1][2] = 0.0;
    filterY[2][2] = 1.0;


    buffer = image->copy();
    for(r=1;r<h-1;r++)
    {
        for(c=1;c<w-1;c++)
        {
            double dx, dy;
            dx = dy = 0.0;

            for(rd=-1;rd<=1;rd++)
                for(cd=-1;cd<=1;cd++)
                {
                    pixel = buffer.pixel(c+cd, r+rd);
                    dx += filterX[rd+1][cd+1]*(double) qGreen(pixel);
                    dy += filterY[rd+1][cd+1]*(double) qGreen(pixel);
                }

            dx /= 8.0;
            dy /= 8.0;

            double mag = min(255.0, 4.0*sqrt(dx*dx + dy*dy));


            image->setPixel(c, r, qRgb( (int) (mag), (int) (mag), (int) (mag)));

        }
    }
}

void MainWindow::OrientationImage(QImage *image)
{
    QImage buffer;
    int w = image->width();
    int h = image->height();
    int r, c, rd, cd;
    QRgb pixel;

    double filterX[3][3];
    double filterY[3][3];

    filterX[0][0] = -1.0;
    filterX[1][0] = -2.0;
    filterX[2][0] = -1.0;

    filterX[0][1] = 0.0;
    filterX[1][1] = 0.0;
    filterX[2][1] = 0.0;

    filterX[0][2] = 1.0;
    filterX[1][2] = 2.0;
    filterX[2][2] = 1.0;

    filterY[0][0] = -1.0;
    filterY[1][0] = 0.0;
    filterY[2][0] = 1.0;

    filterY[0][1] = -2.0;
    filterY[1][1] = 0.0;
    filterY[2][1] = 2.0;

    filterY[0][2] = -1.0;
    filterY[1][2] = 0.0;
    filterY[2][2] = 1.0;


    buffer = image->copy();
    for(r=1;r<h-1;r++)
    {
        for(c=1;c<w-1;c++)
        {
            double dx, dy;
            dx = dy = 0.0;

            for(rd=-1;rd<=1;rd++)
                for(cd=-1;cd<=1;cd++)
                {
                    pixel = buffer.pixel(c+cd, r+rd);
                    dx += filterX[rd+1][cd+1]*(double) qGreen(pixel);
                    dy += filterY[rd+1][cd+1]*(double) qGreen(pixel);
                }

            dx /= 8.0;
            dy /= 8.0;

            double orien = atan2(dy, dx);
            double mag = sqrt(dx*dx + dy*dy);

            double red = (sin(orien) + 1.0)/2.0;
            double green = (cos(orien) + 1.0)/2.0;
            double blue = 1.0 - red - green;

            red *= mag*4.0;
            green *= mag*4.0;
            blue *= mag*4.0;

            red = min(255.0, max(0.0, red));
            green = min(255.0, max(0.0, green));
            blue = min(255.0, max(0.0, blue));

            image->setPixel(c, r, qRgb( (int) (red), (int) (green), (int) (blue)));

        }
    }
}

void MainWindow::BilateralImage(QImage *image, double sigmaS, double sigmaI)
{
    int r, c, rd, cd, i;
    QRgb pixel;
    int radius = max(1, (int) (sigmaS*3.0));
    int size = 2*radius + 1;
    QImage buffer;
    int w = image->width();
    int h = image->height();

    buffer = image->copy();

    if(sigmaS == 0.0)
        return;

    double *kernel = new double [size];

    for(i=0;i<size;i++)
    {
        double dist = (double) (i - radius);

        kernel[i] = exp(-(dist*dist)/(2.0*sigmaS*sigmaS));
    }

    double denom = 0.000001;

    for(i=0;i<size;i++)
        denom += kernel[i];
    for(i=0;i<size;i++)
        kernel[i] /= denom;

    for(r=0;r<h;r++)
    {
        for(c=0;c<w;c++)
        {

            double pixRGB[3];

            pixRGB[0] = 0.0;
            pixRGB[1] = 0.0;
            pixRGB[2] = 0.0;
            double denom = 0.0;
            pixel = buffer.pixel(c, r);

            double inten0 = (double) (qRed(pixel) + qGreen(pixel) + qBlue(pixel))/3.0;

            for(rd=-radius;rd<=radius;rd++)
                for(cd=-radius;cd<=radius;cd++)
                    if(r + rd >= 0 && r + rd < h && c + cd >= 0 && c + cd < w)
                {
                     pixel = buffer.pixel(c+cd, r+rd);
                     double weight = kernel[rd + radius]*kernel[cd + radius];

                     double inten1 = (double) (qRed(pixel) + qGreen(pixel) + qBlue(pixel))/3.0;

                     weight *= exp(-((inten0 - inten1)*(inten0 - inten1))/(2.0*sigmaI*sigmaI));

                     pixRGB[0] += weight*(double) qRed(pixel);
                     pixRGB[1] += weight*(double) qGreen(pixel);
                     pixRGB[2] += weight*(double) qBlue(pixel);
                     denom += weight;
                }

            pixRGB[0] /= denom;
            pixRGB[1] /= denom;
            pixRGB[2] /= denom;

            image->setPixel(c, r, qRgb( (int) pixRGB[0], (int) pixRGB[1], (int) pixRGB[2]));
        }
    }


    delete [] kernel;

}

void MainWindow::BilinearInterpolation(QImage *image, double x, double y, double rgb[3])
{
    int r = (int) y;
    int c = (int) x;
    double rdel = y - (double) r;
    double cdel = x - (double) c;
    QRgb pixel;
    double del;

    rgb[0] = rgb[1] = rgb[2] = 0.0;

    if(r >= 0 && r < image->height() - 1 && c >= 0 && c < image->width() - 1)
    {
        pixel = image->pixel(c, r);
        del = (1.0 - rdel)*(1.0 - cdel);
        rgb[0] += del*(double) qRed(pixel);
        rgb[1] += del*(double) qGreen(pixel);
        rgb[2] += del*(double) qBlue(pixel);

        pixel = image->pixel(c+1, r);
        del = (1.0 - rdel)*(cdel);
        rgb[0] += del*(double) qRed(pixel);
        rgb[1] += del*(double) qGreen(pixel);
        rgb[2] += del*(double) qBlue(pixel);

        pixel = image->pixel(c, r+1);
        del = (rdel)*(1.0 - cdel);
        rgb[0] += del*(double) qRed(pixel);
        rgb[1] += del*(double) qGreen(pixel);
        rgb[2] += del*(double) qBlue(pixel);

        pixel = image->pixel(c+1, r+1);
        del = (rdel)*(cdel);
        rgb[0] += del*(double) qRed(pixel);
        rgb[1] += del*(double) qGreen(pixel);
        rgb[2] += del*(double) qBlue(pixel);
    }
}


void MainWindow::RotateImage(QImage *image, double orien)
{
    int r, c;
    QRgb pixel;
    QImage buffer;
    int w = image->width();
    int h = image->height();
    double radians = -2.0*3.141*orien/360.0;

    buffer = image->copy();

    pixel = qRgb(0, 0, 0);
    image->fill(pixel);

    for(r=0;r<h;r++)
    {
        for(c=0;c<w;c++)
        {
            double rgb[3];
            double x0, y0;
            double x1, y1;

            x0 = (double) (c - w/2);
            y0 = (double) (r - h/2);

            x1 = x0*cos(radians) - y0*sin(radians);
            y1 = x0*sin(radians) + y0*cos(radians);

            x1 += (double) (w/2);
            y1 += (double) (h/2);

            BilinearInterpolation(&buffer, x1, y1, rgb);

            image->setPixel(c, r, qRgb( (int) round(rgb[0]), (int) round(rgb[1]), (int) round(rgb[2])));
        }
    }

}

void MainWindow::FindPeaksImage(QImage *image, double thres)
{
    QImage buffer;
    int w = image->width();
    int h = image->height();
    int r, c, rd, cd;
    QRgb pixel;
    QImage orienImage;
    QImage magImage;

    double filterX[3][3];
    double filterY[3][3];

    filterX[0][0] = -1.0;
    filterX[1][0] = -2.0;
    filterX[2][0] = -1.0;

    filterX[0][1] = 0.0;
    filterX[1][1] = 0.0;
    filterX[2][1] = 0.0;

    filterX[0][2] = 1.0;
    filterX[1][2] = 2.0;
    filterX[2][2] = 1.0;

    filterY[0][0] = -1.0;
    filterY[1][0] = 0.0;
    filterY[2][0] = 1.0;

    filterY[0][1] = -2.0;
    filterY[1][1] = 0.0;
    filterY[2][1] = 2.0;

    filterY[0][2] = -1.0;
    filterY[1][2] = 0.0;
    filterY[2][2] = 1.0;


    buffer = image->copy();
    magImage = image->copy();
    orienImage = image->copy();

    for(r=1;r<h-1;r++)
    {
        for(c=1;c<w-1;c++)
        {
            double dx, dy;
            dx = dy = 0.0;

            for(rd=-1;rd<=1;rd++)
                for(cd=-1;cd<=1;cd++)
                {
                    pixel = buffer.pixel(c+cd, r+rd);
                    dx += filterX[rd+1][cd+1]*(double) qGreen(pixel);
                    dy += filterY[rd+1][cd+1]*(double) qGreen(pixel);
                }

            dx /= 8.0;
            dy /= 8.0;

            double orien = atan2(dy, dx);
            double mag = sqrt(dx*dx + dy*dy);

            magImage.setPixel(c, r, qRgb( 0, (int) (mag), 0));
            int isin = (int) (255.0*((sin(orien) + 1.0)/2.0));
            int icos = (int) (255.0*((cos(orien) + 1.0)/2.0));
            orienImage.setPixel(c, r, qRgb( isin, icos, 0));
        }
    }

    pixel = qRgb(0, 0, 0);
    image->fill(pixel);

    for(r=1;r<h-1;r++)
    {
        for(c=1;c<w-1;c++)
        {
            pixel = magImage.pixel(c, r);
            double mag0 = (double) qGreen(pixel);

            if(mag0 > thres)
            {
                pixel = orienImage.pixel(c, r);
                double dsin = (double) qRed(pixel);
                double dcos = (double) qGreen(pixel);
                dsin /= 255.0;
                dcos /= 255.0;
                dsin *= 2.0;
                dcos *= 2.0;
                dsin--;
                dcos--;

                double x1, y1;
                double rgb1[3];

                x1 = dcos;
                y1 = dsin;
                x1 += (double) c;
                y1 += (double) r;

                BilinearInterpolation(&magImage, x1, y1, rgb1);

                double x2, y2;
                double rgb2[3];

                x2 = -dcos;
                y2 = -dsin;
                x2 += (double) c;
                y2 += (double) r;

                BilinearInterpolation(&magImage, x2, y2, rgb2);

                if(mag0 >= rgb1[1] && mag0 >= rgb2[1])
                    image->setPixel(c, r, qRgb(255, 255, 255));

            }
        }
    }
}


void MainWindow::MedianImage(QImage *image, int radius)
{
    // Add your code here
}

void MainWindow::HoughImage(QImage *image)
{
    // Add your code here
}
