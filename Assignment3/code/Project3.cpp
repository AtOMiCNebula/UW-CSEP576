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

    // First forward project the disparity values
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
                    // Make sure the depth ordering is correct
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
                    // Make sure the depth ordering is correct
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

    // Fill in small holes after the forward projection
    FillHoles(projDisparity, projDisparityCt, w, h);

    renderImage->fill(qRgb(0,0,0));

    // Backward project to find the color values for each pixel
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
    int numDisparities = (maxDisparity - minDisparity); // falls short by one, but this is how the harness determines it

    for (int y = 0; y < h; y++)
    {
        for (int x = 0; x < w; x++)
        {
            QRgb pixelLeft = image1.pixel(x, y);

            for (int d = 0; d < numDisparities; d++)
            {
                int xd = (x - (minDisparity + d));
                QRgb pixelRight;
                if (0 <= xd && xd < w)
                {
                    pixelRight = image2.pixel(xd, y);
                }
                else
                {
                    pixelRight = qRgb(0, 0, 0);
                }

                matchCost[d*w*h + y*w + x] = (pow(0.3*(qRed(pixelLeft) - qRed(pixelRight)), 2) +
                                              pow(0.6*(qGreen(pixelLeft) - qGreen(pixelRight)), 2) +
                                              pow(0.1*(qBlue(pixelLeft) - qBlue(pixelRight)), 2));
            }
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
   int w = image1.width();
   int h = image1.height();
    int numDisparities = (maxDisparity - minDisparity); // falls short by one, but this is how the harness determines it

    for (int y = 0; y < h; y++)
    {
        for (int x = 0; x < w; x++)
        {
            QRgb pixelLeft = image1.pixel(x, y);

            for (int d = 0; d < numDisparities; d++)
            {
                int xd = (x - (minDisparity + d));
                QRgb pixelRight;
                if (0 <= xd && xd < w)
                {
                    pixelRight = image2.pixel(xd, y);
                }
                else
                {
                    pixelRight = qRgb(0, 0, 0);
                }

                matchCost[d*w*h + y*w + x] = abs(0.3*(qRed(pixelLeft) - qRed(pixelRight))) +
                                             abs(0.6*(qGreen(pixelLeft) - qGreen(pixelRight))) +
                                             abs(0.1*(qBlue(pixelLeft) - qBlue(pixelRight)));
            }
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
   int w = image1.width();
   int h = image1.height();

   // Add your code here

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
    // Add your code here
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
    // Add your code here
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
    // Add your code here
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
    // Add your code here
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
    // Add your code here
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
    // Add your code here
}

/*******************************************************************************
For each pixel find the disparity with minimum match cost
    matchCost - The match cost between pixels
    disparities - The disparity for each pixel (use disparity[r*w + c])
    width, height - Width and height of image
    minDisparity - The minimum disparity
    numDisparities - Number of disparities
*******************************************************************************/
void MainWindow::FindBestDisparity(double *matchCost, double *disparities, int w, int h, int minDisparity, int numDisparities)
{
    for (int y = 0; y < h; y++)
    {
        for (int x = 0; x < w; x++)
        {
            int bestDisparity = -1;
            double bestDisparityError = 0.0;

            for (int d = 0; d < numDisparities; d++)
            {
                if (bestDisparity == -1 || matchCost[d*w*h + y*w + x] < bestDisparityError)
                {
                    bestDisparity = d;
                    bestDisparityError = matchCost[d*w*h + y*w + x];
                }
            }

            disparities[y*w+x] = (minDisparity + bestDisparity);
        }
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
    // Add your code here

}
