#include "mainwindow.h"
#include "math.h"
#include "ui_mainwindow.h"
#include <QtGui>

/*******************************************************************************
    The following are helper routines with code already written.
    The routines you'll need to write for the assignment are below.
*******************************************************************************/

/*******************************************************************************
   K-means segment the image
        image - Input image
        gridSize - Initial size of the segments
        numIterations - Number of iterations to run k-means
        spatialSigma - Spatial sigma for measuring distance
        colorSigma - Color sigma for measuring distance
        matchCost - The match cost for each pixel at each disparity
        numDisparities - Number of disparity levels
        segmentImage - Image showing segmentations

*******************************************************************************/
void MainWindow::Segment(QImage image, int gridSize, int numIterations, double spatialSigma, double colorSigma,
                         double *matchCost, int numDisparities, QImage *segmentImage)
{
    int w = image.width();
    int h = image.height();
    int iter;
    int numSegments = 0;

    // Stores the segment assignment for each pixel
    int *segment = new int [w*h];

    // Compute an initial segmentation
    GridSegmentation(segment, numSegments, gridSize, w, h);

    // allocate memory for storing the segments mean position and color
    double (*meanSpatial)[2] = new double [numSegments][2];
    double (*meanColor)[3] = new double [numSegments][3];

    // Iteratively update the segmentation
    for(iter=1;iter<numIterations;iter++)
    {
        // Compute new means
        ComputeSegmentMeans(image, segment, numSegments, meanSpatial, meanColor);
        // Compute new pixel assignment to pixels
        AssignPixelsToSegments(image, segment, numSegments, meanSpatial, meanColor, spatialSigma, colorSigma);
    }

    // Update means again for display
    ComputeSegmentMeans(image, segment, numSegments, meanSpatial, meanColor);
    // Display the segmentation
    DrawSegments(segmentImage, segment, meanColor);

    // Update the match cost based on the segmentation
    SegmentAverageMatchCost(segment, numSegments, w, h, numDisparities, matchCost);

    delete [] meanSpatial;
    delete [] meanColor;
    delete [] segment;
}

/*******************************************************************************
   Compute initial segmentation of the image using a grid
        segment - Segment assigned to each pixel
        numSegments - Number of segments
        gridSize - Size of the grid-based segments
        w - Image width
        h - Image height

*******************************************************************************/
void MainWindow::GridSegmentation(int *segment, int &numSegments, int gridSize, int w, int h)
{
    int r, c;
    int step = w/gridSize;

    if(step*gridSize < w)
        step += 1;

    for(r=0;r<h;r++)
        for(c=0;c<w;c++)
        {
             int rs = r/gridSize;
             int cs = c/gridSize;

             segment[r*w + c] = rs*step + cs;

             numSegments = rs*step + cs + 1;

        }

}

/*******************************************************************************
   Draw the image segmentation
        segmentImage - Image to display the segmentation
        segment - Segment assigned to each pixel
        meanColor - The mean color of the segments

*******************************************************************************/
void MainWindow::DrawSegments(QImage *segmentImage, int *segment, double (*meanColor)[3])
{
    int w = segmentImage->width();
    int h = segmentImage->height();
    int r, c;

    for(r=0;r<h-1;r++)
        for(c=0;c<w-1;c++)
        {
            int segIdx = segment[r*w + c];
            if(segIdx != segment[r*w + c + 1] ||
               segIdx != segment[(r+1)*w + c])
            {
                segmentImage->setPixel(c, r, qRgb(255, 255, 255));
            }
            else
            {
                segmentImage->setPixel(c, r, qRgb((int) meanColor[segIdx][0],
                                                  (int) meanColor[segIdx][1], (int) meanColor[segIdx][2]));
            }
        }
}

/*******************************************************************************
   Display the computed disparities
        disparities - The disparity for each pixel
        disparityScale - The amount to scale the disparity for display
        minDisparity - Minimum disparity
        disparityImage - Image to display the disparity
        errorImage - Image to display the error
        GTImage - The ground truth disparities
        m_DisparityError - The average error

*******************************************************************************/
void MainWindow::DisplayDisparities(double *disparities, int disparityScale, int minDisparity,
                        QImage *disparityImage, QImage *errorImage, QImage GTImage, double *disparityError)
{
    int w = disparityImage->width();
    int h = disparityImage->height();
    int r, c;
    int gtw = GTImage.width();
    bool useGT = false;
    double pixelCt = 0.0;
    *disparityError = 0.0;
    double maxError = 1.0*(double) disparityScale;

    if(gtw == w)
        useGT = true;

    QRgb pixel;

    for(r=0;r<h;r++)
        for(c=0;c<w;c++)
        {
            double disparity = disparities[r*w + c];
            disparity *= (double) disparityScale;
            disparity -= minDisparity*disparityScale;

            disparityImage->setPixel(c, r, qRgb((int) disparity, (int) disparity, (int) disparity));

            if(useGT)
            {
                pixel = GTImage.pixel(c, r);

                if(qGreen(pixel) > 0)
                {
                    double dist = fabs(disparity - (double) qGreen(pixel));
                    if(dist > maxError)
                        (*disparityError)++;
                    pixelCt++;

                    if(dist > maxError)
                        errorImage->setPixel(c, r, qRgb(255,255,255));
                    else
                        errorImage->setPixel(c, r, qRgb(0,0,0));
                }


            }
        }

    if(useGT)
        *disparityError /= pixelCt;
}

/*******************************************************************************
   Render warped views between the images
        image - Image to be warped
        disparities - The disparities for each pixel
        disparityScale - The amount to warp the image, usually between 0 and 1
        renderImage - The final rendered image

*******************************************************************************/
void MainWindow::Render(QImage image, double *disparities, double disparityScale, QImage *renderImage)
{
    int r, c;
    int w = image.width();
    int h = image.height();
    double *projDisparity = new double [w*h];
    double *projDisparityCt = new double [w*h];
    QRgb pixel0;
    QRgb pixel1;

    memset(projDisparity, 0, w*h*sizeof(double));
    memset(projDisparityCt, 0, w*h*sizeof(double));

    for(r=0;r<h;r++)
        for(c=0;c<w;c++)
        {
            double disparity =  -disparities[r*w + c]*disparityScale;
            double x = (double) c + disparity;
            int cp = (int) x;
            double del = x - (double) cp;

            if(cp >= 0 && cp < w-1)
            {
                // Make sure we get the depth ordering correct.
                if(projDisparityCt[r*w + cp] == 0)
                {
                    projDisparity[r*w + cp] = (1.0 - del)*disparity;
                    projDisparityCt[r*w + cp] = (1.0 - del);
                }
                else
                {
                    if(fabs(disparity) > fabs(2.0 + projDisparity[r*w + cp]/projDisparityCt[r*w + cp]))
                    {
                        projDisparity[r*w + cp] = (1.0 - del)*disparity;
                        projDisparityCt[r*w + cp] = (1.0 - del);
                    }
                    else
                    {
                        projDisparity[r*w + cp] += (1.0 - del)*disparity;
                        projDisparityCt[r*w + cp] += (1.0 - del);
                    }
                }

                if(projDisparityCt[r*w + cp + 1] == 0)
                {
                    projDisparity[r*w + cp + 1] = (del)*disparity;
                    projDisparityCt[r*w + cp + 1] = (del);
                }
                else
                {
                    if(fabs(disparity) > fabs(2.0 + projDisparity[r*w + cp + 1]/projDisparityCt[r*w + cp + 1]))
                    {
                        projDisparity[r*w + cp + 1] = (del)*disparity;
                        projDisparityCt[r*w + cp + 1] = (del);
                    }
                    else
                    {
                        projDisparity[r*w + cp + 1] += (del)*disparity;
                        projDisparityCt[r*w + cp + 1] += (del);
                    }
                }
            }
        }

    for(r=0;r<h;r++)
        for(c=0;c<w;c++)
        {
            if(projDisparityCt[r*w + c] > 0.0)
            {
                projDisparity[r*w + c] /= projDisparityCt[r*w + c];
            }
        }

    FillHoles(projDisparity, projDisparityCt, w, h);

    renderImage->fill(qRgb(0,0,0));

    for(r=0;r<h;r++)
        for(c=0;c<w;c++)
            if(projDisparityCt[r*w + c] > 0.0)
        {
            double disparity =  projDisparity[r*w + c];
            double x = (double) c - disparity;
            int cp = (int) x;
            double del = x - (double) cp;

            if(cp >= 0 && cp < w-1)
            {
                pixel0 = image.pixel(cp, r);
                pixel1 = image.pixel(cp+1, r);

                int red = (int) ((1.0 - del)*(double)qRed(pixel0) + del*(double)qRed(pixel1));
                int green = (int) ((1.0 - del)*(double)qGreen(pixel0) + del*(double)qGreen(pixel1));
                int blue = (int) ((1.0 - del)*(double)qBlue(pixel0) + del*(double)qBlue(pixel1));

                // Uncomment if you want to see the disparities
            //    red = (int) disparity*4.0;
            //    green = (int) disparity*4.0;
            //    blue = (int) disparity*4.0;

                renderImage->setPixel(c, r, qRgb(red, green, blue));
            }
        }


    delete [] projDisparity;
    delete [] projDisparityCt;
}

/*******************************************************************************
   Fill holes in the projected disparities (Render helper function)
        projDisparity - Projected disparity
        projDisparityCt - The weight of each projected disparity.  A value of 0 means the pixel doesn't have a disparity
        w, h - The width and height of the image

*******************************************************************************/
void MainWindow::FillHoles(double *projDisparity, double *projDisparityCt, int w, int h)
{
    int r, c, cd, rd;
    double *bufferCt = new double [w*h];

    memcpy(bufferCt, projDisparityCt, w*h*sizeof(double));

    for(r=1;r<h-1;r++)
        for(c=1;c<w-1;c++)
            if(bufferCt[r*w + c] == 0)
        {
            double avgDisparity = 0.0;
            double avgCt = 0.0;

            for(rd=-1;rd<=1;rd++)
                for(cd=-1;cd<=1;cd++)
                {
                    int idx = (r + rd)*w + c + cd;
                   avgDisparity += projDisparity[idx]*bufferCt[idx];
                   avgCt += bufferCt[idx];
                }

            if(avgCt > 0.0)
            {
                projDisparity[r*w + c] = avgDisparity/avgCt;
                projDisparityCt[r*w + c] = avgCt;

            }
        }

    delete [] bufferCt;
}


/*******************************************************************************
*******************************************************************************
*******************************************************************************

    The routines you need to implement are below

*******************************************************************************
*******************************************************************************
*******************************************************************************/

/*******************************************************************************
Compute match cost using Squared Distance
    image1 - Input image 1
    image2 - Input image 2
    minDisparity - Minimum disparity between image 1 and image 2
    maxDisparity - Minimum disparity between image 1 and image 2
    matchCost - The match cost (squared distance) between pixels

    To access the match cost at pixel (c, r) at disparity d use
    matchCost[d*w*h + r*w + c]
*******************************************************************************/
void MainWindow::SSD(QImage image1, QImage image2, int minDisparity, int maxDisparity, double *matchCost)
{
    int w = image1.width();
    int h = image1.height();

    int r1, c1, c2, d;
    QRgb pixel1;
    QRgb pixel2;

    for(d=minDisparity;d<maxDisparity;d++)
    {
       int idx = (d - minDisparity)*w*h;

        for(r1=0;r1<h;r1++)
            for(c1=0;c1<w;c1++)
            {
                c2 = c1 - d;

                if(c2 >=0 && c2 < w)
                {
                    pixel1 = image1.pixel(c1, r1);
                    pixel2 = image2.pixel(c2, r1);

                    float diffr = (float) qRed(pixel1) - (float) qRed(pixel2);
                    float diffg = (float) qGreen(pixel1) - (float) qGreen(pixel2);
                    float diffb = (float) qBlue(pixel1) - (float) qBlue(pixel2);

                    matchCost[idx + r1*w + c1] = sqrt(diffr*diffr + diffg*diffg + diffb*diffb);

                }
                else
                    matchCost[idx + r1*w + c1] = 0.0;
            }
    }

}

/*******************************************************************************
Compute match cost using Absolute Distance
    image1 - Input image 1
    image2 - Input image 2
    minDisparity - Minimum disparity between image 1 and image 2
    maxDisparity - Minimum disparity between image 1 and image 2
    matchCost - The match cost (absolute distance) between pixels

    To access the match cost at pixel (c, r) at disparity d use
    matchCost[d*w*h + r*w + c]
*******************************************************************************/
void MainWindow::SAD(QImage image1, QImage image2, int minDisparity, int maxDisparity, double *matchCost)
{
   int r1, c1, c2, d;
   int w = image1.width();
   int h = image1.height();
   QRgb pixel1;
   QRgb pixel2;

   for(d=minDisparity;d<maxDisparity;d++)
   {
       int idx = (d - minDisparity)*w*h;

        for(r1=0;r1<h;r1++)
            for(c1=0;c1<w;c1++)
            {
                c2 = c1 - d;

                if(c2 >=0 && c2 < w)
                {
                    pixel1 = image1.pixel(c1, r1);
                    pixel2 = image2.pixel(c2, r1);

                    float diffr = (float) qRed(pixel1) - (float) qRed(pixel2);
                    float diffg = (float) qGreen(pixel1) - (float) qGreen(pixel2);
                    float diffb = (float) qBlue(pixel1) - (float) qBlue(pixel2);

                    matchCost[idx + r1*w + c1] = fabs(diffr) + fabs(diffg) + fabs(diffb);

                }
                else
                    matchCost[idx + r1*w + c1] = 0.0;
            }
   }

}

/*******************************************************************************
Compute match cost using Normalized Cross Correlation
    image1 - Input image 1
    image2 - Input image 2
    minDisparity - Minimum disparity between image 1 and image 2
    maxDisparity - Minimum disparity between image 1 and image 2
    radius - Radius of window to compute the NCC score
    matchCost - The match cost (1 - NCC) between pixels

    To access the match cost at pixel (c, r) at disparity d use
    matchCost[d*w*h + r*w + c]
*******************************************************************************/
void MainWindow::NCC(QImage image1, QImage image2, int minDisparity, int maxDisparity, int radius, double *matchCost)
{
   int r1, c1, c2, d, rd, cd;
   int w = image1.width();
   int h = image1.height();
   QRgb pixel1;
   QRgb pixel2;

   for(d=minDisparity;d<maxDisparity;d++)
   {
       int idx = (d - minDisparity)*w*h;

        for(r1=radius;r1<h-radius;r1++)
            for(c1=radius;c1<w-radius;c1++)
            {
                c2 = c1 - d;

                if(c2 >= radius && c2 < w - radius)
                {
                    double xx, xy, yy;
                    xx = xy = yy = 0.00001;

                    for(rd=-radius;rd<=radius;rd++)
                        for(cd=-radius;cd<=radius;cd++)
                        {
                            pixel1 = image1.pixel(c1 + cd, r1 + rd);
                            pixel2 = image2.pixel(c2 + cd, r1 + rd);

                            double r1 = (double) qRed(pixel1);
                            double g1 = (double) qGreen(pixel1);
                            double b1 = (double) qBlue(pixel1);

                            double r2 = (double) qRed(pixel2);
                            double g2 = (double) qGreen(pixel2);
                            double b2 = (double) qBlue(pixel2);

                            xx += r1*r1;
                            xx += g1*g1;
                            xx += b1*b1;

                            xy += r1*r2;
                            xy += g1*g2;
                            xy += b1*b2;

                            yy += r2*r2;
                            yy += g2*g2;
                            yy += b2*b2;
                        }

                    matchCost[idx + r1*w + c1] = 1.0 - xy/(sqrt(xx*yy));

                }
                else
                    matchCost[idx + r1*w + c1] = 0.0;
            }
   }

}

/*******************************************************************************
Gaussian blur the match score.
    matchCost - The match cost between pixels
    w, h - The width and height of the image
    numDisparities - The number of disparity levels
    sigma - The standard deviation of the blur kernel

    I would recommend using SeparableGaussianBlurImage as a helper function.
*******************************************************************************/
void MainWindow::GaussianBlurMatchScore(double *matchCost, int w, int h, int numDisparities, double sigma)
{
    int d;

    for(d=0;d<numDisparities;d++)
        SeparableGaussianBlurImage(&(matchCost[d*w*h]), w, h, sigma);
}

/*******************************************************************************
Blur a floating piont image using Gaussian kernel (helper function for GaussianBlurMatchScore.)
    image - Floating point image
    w, h - The width and height of the image
    sigma - The standard deviation of the blur kernel

    You may just cut and paste code from previous assignment
*******************************************************************************/
void MainWindow::SeparableGaussianBlurImage(double *image, int w, int h, double sigma)
{
    int r, c, rd, cd, i;
    int radius = max(1, (int) (sigma*3.0));
    int size = 2*radius + 1;
    double *buffer = new double [w*h];

    memcpy(buffer, image, w*h*sizeof(double));

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
            double val = 0.0;
            double denom = 0.0;

            for(rd=-radius;rd<=radius;rd++)
                if(r + rd >= 0 && r + rd < h)
                {
                     double weight = kernel[rd + radius];

                     val += weight*buffer[(r + rd)*w + c];
                     denom += weight;
                }

            val /= denom;

            image[r*w + c] = val;
        }
    }

    memcpy(buffer, image, w*h*sizeof(double));

    for(r=0;r<h;r++)
    {
        for(c=0;c<w;c++)
        {
            double val = 0.0;
            double denom = 0.0;

            for(cd=-radius;cd<=radius;cd++)
                if(c + cd >= 0 && c + cd < w)
                {
                     double weight = kernel[cd + radius];

                     val += weight*buffer[r*w + c + cd];
                     denom += weight;
                }

            val /= denom;

            image[r*w + c] = val;
        }
    }


    delete [] kernel;
    delete [] buffer;
}



/*******************************************************************************
Bilaterally blur the match score using the colorImage to compute kernel weights
    matchCost - The match cost between pixels
    w, h - The width and height of the image
    numDisparities - The number of disparity levels
    sigmaS, sigmaI - The standard deviation of the blur kernel for spatial and intensity
    colorImage - The color image
*******************************************************************************/
void MainWindow::BilateralBlurMatchScore(double *matchCost, int w, int h, int numDisparities,
                                         double sigmaS, double sigmaI, QImage colorImage)
{
    int d;

    int r, c, rd, cd, i;
    QRgb pixel;
    int radius = max(1, (int) (sigmaS*3.0));
    int size = 2*radius + 1;
    double  *buffer;

    buffer = new double [w*h*numDisparities];
    memcpy(buffer, matchCost, w*h*numDisparities*sizeof(double));
    memset(matchCost, 0, w*h*numDisparities*sizeof(double));

    if(sigmaS == 0.0)
        return;

    double *kernel = new double [size];

    for(i=0;i<size;i++)
    {
        double dist = (double) (i - radius);

        kernel[i] = exp(-(dist*dist)/(2.0*sigmaS*sigmaS));
    }

    for(r=radius;r<h-radius;r++)
    {
        for(c=radius;c<w-radius;c++)
        {
            double denom = 0.000001;

            pixel = colorImage.pixel(c, r);
            double inten0 = (double) qGreen(pixel);

            for(rd=-radius;rd<=radius;rd++)
                for(cd=-radius;cd<=radius;cd++)
                {
                     pixel = colorImage.pixel(c+cd, r+rd);
                     double weight = kernel[rd + radius]*kernel[cd + radius];

                     double inten1 = qGreen(pixel);

                     weight *= exp(-((inten0 - inten1)*(inten0 - inten1))/(2.0*sigmaI*sigmaI));

                     for(d=0;d<numDisparities;d++)
                        matchCost[d*h*w + r*w + c] += weight*(double) buffer[d*w*h + (r+rd)*w + c + cd];

                     denom += weight;
                }

            for(d=0;d<numDisparities;d++)
                matchCost[d*h*w + r*w + c] /= denom;
        }
    }


    delete [] buffer;
    delete [] kernel;

}

/*******************************************************************************
Compute the mean color and position for each segment (helper function for Segment.)
    image - Color image
    segment - Image segmentation
    numSegments - Number of segments
    meanSpatial - Mean position of segments
    meanColor - Mean color of segments
*******************************************************************************/
void MainWindow::ComputeSegmentMeans(QImage image, int *segment, int numSegments, double (*meanSpatial)[2], double (*meanColor)[3])
{
    double *meanCt = new double [numSegments];
    int r, c, i;
    int w = image.width();
    int h = image.height();
    QRgb pixel;

    memset(meanCt, 0, numSegments*sizeof(double));

    memset(meanSpatial, 0, 2*numSegments*sizeof(double));
    memset(meanColor, 0, 3*numSegments*sizeof(double));

    for(r=0;r<h;r++)
        for(c=0;c<w;c++)
        {
            int segIdx = segment[r*w + c];
            pixel = image.pixel(c, r);

            meanSpatial[segIdx][0] += (double) c;
            meanSpatial[segIdx][1] += (double) r;
            meanCt[segIdx]++;

            meanColor[segIdx][0] += (double) qRed(pixel);
            meanColor[segIdx][1] += (double) qGreen(pixel);
            meanColor[segIdx][2] += (double) qBlue(pixel);
        }

    for(i=0;i<numSegments;i++)
    {
        if(meanCt[i] > 0.0)
        {
            meanSpatial[i][0] /= meanCt[i];
            meanSpatial[i][1] /= meanCt[i];

            meanColor[i][0] /= meanCt[i];
            meanColor[i][1] /= meanCt[i];
            meanColor[i][2] /= meanCt[i];
        }
    }

    delete [] meanCt;

}

/*******************************************************************************
Assign each pixel to the closest segment using position and color
    image - Color image
    segment - Image segmentation
    numSegments - Number of segments
    meanSpatial - Mean position of segments
    meanColor - Mean color of segments
    spatialSigma - Assumed standard deviation of the spatial distribution of pixels in segment
    colorSigma - Assumed standard deviation of the color distribution of pixels in segment
*******************************************************************************/
void MainWindow::AssignPixelsToSegments(QImage image, int *segment, int numSegments, double (*meanSpatial)[2], double (*meanColor)[3],
                            double spatialSigma, double colorSigma)
{
    int r, c, i;
    int w = image.width();
    int h = image.height();
    QRgb pixel;

    for(r=0;r<h;r++)
        for(c=0;c<w;c++)
        {
            double minDist = 99999999.0;
            int minIdx;

            pixel = image.pixel(c, r);

            for(i=0;i<numSegments;i++)
            {
                double distS;
                double distC;



                distS = ((double) c - meanSpatial[i][0])*((double) c - meanSpatial[i][0]) +
                        ((double) r - meanSpatial[i][1])*((double) r - meanSpatial[i][1]);

                distC = ((double) qRed(pixel) - meanColor[i][0])*((double) qRed(pixel) - meanColor[i][0]) +
                        ((double) qGreen(pixel) - meanColor[i][1])*((double) qGreen(pixel) - meanColor[i][1]) +
                        ((double) qBlue(pixel) - meanColor[i][2])*((double) qBlue(pixel) - meanColor[i][2]);

                distS /= spatialSigma*spatialSigma;
                distC /= colorSigma*colorSigma;

                if(distS + distC < minDist)
                {
                    minDist = distS + distC;
                    minIdx = i;
                }
            }

            segment[r*w + c] = minIdx;
        }
}

/*******************************************************************************
Update the match cost based ont eh segmentation.  That is, average the match cost
for each pixel in a segment.
    segment - Image segmentation
    numSegments - Number of segments
    width, height - Width and height of image
    numDisparities - Number of disparities
    matchCost - The match cost between pixels
*******************************************************************************/
void MainWindow::SegmentAverageMatchCost(int *segment, int numSegments,
                                         int w, int h, int numDisparities, double *matchCost)
{
    int r, c, d;
    double *avgMatchCost = new double [numSegments];
    double *avgCt = new double [numSegments];


    for(d=0;d<numDisparities;d++)
    {
        memset(avgMatchCost, 0, numSegments*sizeof(double));
        memset(avgCt, 0, numSegments*sizeof(double));

        for(r=0;r<h;r++)
            for(c=0;c<w;c++)
            {
                int segIdx = segment[r*w + c];

                avgMatchCost[segIdx] += matchCost[d*w*h + r*w + c];
                avgCt[segIdx]++;
            }

        for(r=0;r<h;r++)
            for(c=0;c<w;c++)
            {
                int segIdx = segment[r*w + c];
                matchCost[d*w*h + r*w + c] = avgMatchCost[segIdx]/avgCt[segIdx];
            }
    }

    delete [] avgMatchCost;
    delete [] avgCt;
}

/*******************************************************************************
For each pixel find the disparity with minimum match cost
    matchCost - The match cost between pixels
    disparities - The disparity for each pixel disparities[r*w + c]
    width, height - Width and height of image
    minDisparity - The minimum disparity
    numDisparities - Number of disparities
*******************************************************************************/
void MainWindow::FindBestDisparity(double *matchCost, double *disparities, int w, int h, int minDisparity, int numDisparities)
{
    int r ,c, d;

    for(r=0;r<h;r++)
        for(c=0;c<w;c++)
        {
            double minCost = 99999999.0;
            int minIdx = 0;

            for(d=0;d<numDisparities;d++)
            {
                if(minCost > matchCost[d*w*h + r*w + c])
                {
                    minCost = matchCost[d*w*h + r*w + c];
                    minIdx = d + minDisparity;
                }

            }

            disparities[r*w + c] = (double) minIdx;

        }
}

/*******************************************************************************
Create your own "magic" stereo algorithm
    image1 - Input image 1
    image2 - Input image 2
    minDisparity - Minimum disparity between image 1 and image 2
    maxDisparity - Minimum disparity between image 1 and image 2
    param1 - The first parameter to your algorithm
    param2 - The second paramater to your algorithm
    matchCost - The match cost (squared distance) between pixels
*******************************************************************************/
void MainWindow::MagicStereo(QImage image1, QImage image2, int minDisparity, int maxDisparity, double param1, double param2, double *matchCost)
{

}
