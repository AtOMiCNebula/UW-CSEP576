#include "mainwindow.h"
#include "math.h"
#include "ui_mainwindow.h"
#include <QtGui>

/***********************************************************************
  This is the only file you need to change for your assignment.  The
  other files control the UI (in case you want to make changes.)
************************************************************************/


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
            double intensity = 0.3*red + 0.6*green + 0.1*blue;

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
            // otherwise add the same amount of noise to each channel
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

    // This is the size of the kernel
    int size = 2*radius + 1;

    // Create a buffer image so we're not reading and writing to the same image during filtering.
    QImage buffer;
    int w = image->width();
    int h = image->height();

    // This creates an image of size (w + 2*radius, h + 2*radius) with black borders.
    // This could be improved by filling the pixels using a different padding technique (reflected, fixed, etc.)
    buffer = image->copy(-radius, -radius, w + 2*radius, h + 2*radius);

    // Compute kernel to convolve with the image.
    double *kernel = new double [size*size];

    for(i=0;i<size*size;i++)
    {
        kernel[i] = 1.0;
    }

    // Make sure kernel sums to 1
    double denom = 0.000001;
    for(i=0;i<size*size;i++)
        denom += kernel[i];
    for(i=0;i<size*size;i++)
        kernel[i] /= denom;

    // For each pixel in the image...
    for(r=0;r<h;r++)
    {
        for(c=0;c<w;c++)
        {
            double rgb[3];

            rgb[0] = 0.0;
            rgb[1] = 0.0;
            rgb[2] = 0.0;

            // Convolve the kernel at each pixel
            for(rd=-radius;rd<=radius;rd++)
                for(cd=-radius;cd<=radius;cd++)
                {
                     // Get the pixel value
                     pixel = buffer.pixel(c + cd + radius, r + rd + radius);

                     // Get the value of the kernel
                     double weight = kernel[(rd + radius)*size + cd + radius];

                     rgb[0] += weight*(double) qRed(pixel);
                     rgb[1] += weight*(double) qGreen(pixel);
                     rgb[2] += weight*(double) qBlue(pixel);
                }

            // Store mean pixel in the image to be returned.
            image->setPixel(c, r, qRgb((int) floor(rgb[0] + 0.5), (int) floor(rgb[1] + 0.5), (int) floor(rgb[2] + 0.5)));
        }
    }

    // Clean up.
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



QRgb ClampPixel(int r, int g, int b)
{
    return qRgb(max(0, min(255, r)),
                max(0, min(255, g)),
                max(0, min(255, b)));
}

void ConvolveHelper(QImage *image, double *kernel, int kernelWidth, int kernelHeight, bool fForDerivative)
{
    int height = image->height();
    int width = image->width();
    int kernelHalfHeight = (kernelHeight / 2);
    int kernelHalfWidth = (kernelWidth / 2);

    QImage buffer = image->copy(-kernelHalfWidth, -kernelHalfHeight, width + 2*kernelHalfWidth, height + 2*kernelHalfHeight );
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            double rgb[3];
            rgb[0] = rgb[1] = rgb[2] = (!fForDerivative ? 0.0 : 128.0);

            for (int fy = 0; fy < kernelHeight; fy++)
            {
                for (int fx = 0; fx < kernelWidth; fx++)
                {
                    // Translate to coordinates in buffer space
                    int by = y + fy;
                    int bx = x + fx;

                    QRgb pixel = buffer.pixel(bx, by);
                    double kernelWeight = kernel[fy*kernelWidth+fx];
                    rgb[0] += kernelWeight*qRed(pixel);
                    rgb[1] += kernelWeight*qGreen(pixel);
                    rgb[2] += kernelWeight*qBlue(pixel);
                }
            }

            image->setPixel(x, y, ClampPixel(static_cast<int>(floor(rgb[0]+0.5)),
                        static_cast<int>(floor(rgb[1]+0.5)),
                        static_cast<int>(floor(rgb[2]+0.5)))
                    );
        }
    }
}

void MainWindow::GaussianBlurImage(QImage *image, double sigma)
{
    if (sigma <= 0)
    {
        return;
    }

    // Calculate the kernel (some extra computation/storage, should be fine...)
    double twoSigSq = 2.0 * pow(sigma, 2);
    int kernelHalfSide = static_cast<int>(ceil(3 * sigma));
    int kernelSize = ((2 * kernelHalfSide) + 1);
    double *kernel = new double[kernelSize * kernelSize];
    for (int i = 0; i < kernelSize; i++)
    {
        int y = (i - kernelHalfSide);
        for (int j = 0; j < kernelSize; j++)
        {
            int x = (j - kernelHalfSide);
            kernel[i*kernelSize+j] = (1.0 / (M_PI*twoSigSq)) * pow(M_E, -1*(pow(x,2.0)+pow(y,2.0))/twoSigSq);
        }
    }

    // Generate the updated image
    ConvolveHelper(image, kernel, kernelSize, kernelSize, false/*fForDerivative*/);

    // Clean up!
    delete[] kernel;
}

void MainWindow::SeparableGaussianBlurImage(QImage *image, double sigma)
{
    if (sigma <= 0)
    {
        return;
    }

    // Calculate the kernel (some extra computation/storage, should be fine...)
    double twoSigSq = 2.0 * pow(sigma, 2);
    double sigSqRt = sigma * sqrt(2*M_PI);
    int kernelHalfSide = static_cast<int>(ceil(3 * sigma));
    int kernelSize = ((2 * kernelHalfSide) + 1);
    double *kernel = new double[kernelSize];
    for (int i = 0; i < kernelSize; i++)
    {
        int y = (i - kernelHalfSide);
        kernel[i] = (1.0 / (sigSqRt)) * pow(M_E, -1*(pow(y,2.0))/twoSigSq);
    }

    // Generate the updated image
    ConvolveHelper(image, kernel, kernelSize, 1, false/*fForDerivative*/);
    ConvolveHelper(image, kernel, 1, kernelSize, false/*fForDerivative*/);

    // Clean up!
    delete[] kernel;
}

void MainWindow::FirstDerivImage(QImage *image, double sigma)
{
    if (sigma <= 0)
    {
        return;
    }

    // Construct the kernel
    double kernel[3] = { -1.0, 0.0, 1.0 };

    // Generate the updated image
    ConvolveHelper(image, kernel, 3, 1, true/*fForDerivative*/);
    GaussianBlurImage(image, sigma);
}

void MainWindow::SecondDerivImage(QImage *image, double sigma)
{
    if (sigma <= 0)
    {
        return;
    }

    // Construct the kernel
    // (This is inverted relative to what the lecture slides suggested,
    // otherwise the colors come out backwards...)
    double kernel[9] = { 0.0,  1.0, 0.0,
                         1.0, -4.0, 1.0,
                         0.0,  1.0, 0.0 };

    // Generate the updated image
    ConvolveHelper(image, kernel, 3, 3, true/*fForDerivative*/);
    GaussianBlurImage(image, sigma);
}

void MainWindow::SharpenImage(QImage *image, double sigma, double alpha)
{
    // Generate the second derivative
    QImage imageSecondDeriv(*image);
    SecondDerivImage(&imageSecondDeriv, sigma);

    // Subtract the second derivative from our original image
    int height = image->height();
    int width = image->width();
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            QRgb pixel = image->pixel(x, y);
            QRgb pixelSecondDeriv = imageSecondDeriv.pixel(x, y);

            image->setPixel(x, y, ClampPixel(
                        qRed(pixel) - alpha*(qRed(pixelSecondDeriv)-128),
                        qGreen(pixel) - alpha*(qGreen(pixelSecondDeriv)-128),
                        qBlue(pixel) - alpha*(qBlue(pixelSecondDeriv)-128)
                    ));
        }
    }
}

void MainWindow::BilateralImage(QImage *image, double sigmaS, double sigmaI)
{
    // Add your code here.  Should be similar to GaussianBlurImage.
}

void MainWindow::SobelImage(QImage *image)
{
    double kernelX[9] = { -1,  0,  1,
                          -2,  0,  2,
                          -1,  0,  1 };
    double kernelY[9] = {  1,  2,  1,
                           0,  0,  0,
                          -1, -2, -1 };

    // We need to convolve two kernels simultaneously to build our result...so
    // I'll duplicate a lot of ConvolveHelper's code here, along with the Sobel
    // helper code.
    QImage buffer = image->copy(-1, -1, image->width()+1, image->height()+1);
    int height = image->height();
    int width = image->width();
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            double rgbX[3] = { 0.0, 0.0, 0.0 };
            double rgbY[3] = { 0.0, 0.0, 0.0 };

            for (int fy = 0; fy < 3; fy++)
            {
                for (int fx = 0; fx < 3; fx++)
                {
                    // Translate to coordinates in buffer space
                    int by = y + fy;
                    int bx = x + fx;

                    QRgb pixel = buffer.pixel(bx, by);
                    double kernelWeightX = kernelX[fy*3+fx];
                    rgbX[0] += kernelWeightX*qRed(pixel);
                    rgbX[1] += kernelWeightX*qGreen(pixel);
                    rgbX[2] += kernelWeightX*qBlue(pixel);
                    double kernelWeightY = kernelY[fy*3+fy];
                    rgbY[0] += kernelWeightY*qRed(pixel);
                    rgbY[1] += kernelWeightY*qGreen(pixel);
                    rgbY[2] += kernelWeightY*qBlue(pixel);
                }
            }

            // Using the intensity function from BlackWhiteImage:
            double intensityX = 0.3*rgbX[0] + 0.6*rgbX[1] + 0.1*rgbX[2];
            double intensityY = 0.3*rgbY[0] + 0.6*rgbY[1] + 0.1*rgbY[2];

            // Provided Sobel helper code
            double mag = sqrt(pow(intensityX, 2) + pow(intensityY, 2));
            double orien = atan2(intensityY, intensityX);
            double red = (sin(orien) + 1.0)/2.0;
            double green = (cos(orien) + 1.0)/2.0;
            double blue = 1.0 - red - green;
            red *= mag*4.0;
            green *= mag*4.0;
            blue *= mag*4.0;

            image->setPixel(x, y, ClampPixel(static_cast<int>(floor(red+0.5)),
                                             static_cast<int>(floor(green+0.5)),
                                             static_cast<int>(floor(blue+0.5))));
        }
    }
}


void MainWindow::BilinearInterpolation(QImage *image, double x, double y, double rgb[3])
{
    int height = image->height();
    int width = image->width();
    int x1 = static_cast<int>(floor(x));
    int y1 = static_cast<int>(floor(y));
    int x2 = static_cast<int>(ceil(x+0.00001));
    int y2 = static_cast<int>(ceil(y+0.00001));

    QRgb pixel11 = ((0 <= x1 && x1 < width && 0 <= y1 && y1 < height) ?
                        image->pixel(x1, y1) : qRgb(0, 0, 0));
    QRgb pixel12 = ((0 <= x1 && x1 < width && 0 <= y2 && y2 < height) ?
                        image->pixel(x1, y2) : qRgb(0, 0, 0));
    QRgb pixel21 = ((0 <= x2 && x2 < width && 0 <= y1 && y1 < height) ?
                        image->pixel(x2, y1) : qRgb(0, 0, 0));
    QRgb pixel22 = ((0 <= x2 && x2 < width && 0 <= y2 && y2 < height) ?
                        image->pixel(x2, y2) : qRgb(0, 0, 0));

    for (int i = 0; i < 3; i++)
    {
        int (*colorfn)(QRgb) = (i == 0 ? qRed : (i == 1 ? qGreen : qBlue));
        rgb[i] = (1 / ((x2-x1)*(y2-y1))) *
                (((*colorfn)(pixel11)*(x2-x)*(y2-y)) +
                 ((*colorfn)(pixel21)*(x-x1)*(y2-y)) +
                 ((*colorfn)(pixel12)*(x2-x)*(y-y1)) +
                 ((*colorfn)(pixel22)*(x-x1)*(y-y1)));
    }
}

// Here is some sample code for rotating an image.  I assume orien is in degrees.

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

            // Rotate around the center of the image.
            x0 = (double) (c - w/2);
            y0 = (double) (r - h/2);

            // Rotate using rotation matrix
            x1 = x0*cos(radians) - y0*sin(radians);
            y1 = x0*sin(radians) + y0*cos(radians);

            x1 += (double) (w/2);
            y1 += (double) (h/2);

            BilinearInterpolation(&buffer, x1, y1, rgb);

            image->setPixel(c, r, qRgb((int) floor(rgb[0] + 0.5), (int) floor(rgb[1] + 0.5), (int) floor(rgb[2] + 0.5)));
        }
    }

}

void MainWindow::FindPeaksImage(QImage *image, double thres)
{
    // Add your code here.
}


void MainWindow::MedianImage(QImage *image, int radius)
{
    // Add your code here
}

void MainWindow::HoughImage(QImage *image)
{
    // Add your code here
}

void MainWindow::CrazyImage(QImage *image)
{
    // Add your code here
}
