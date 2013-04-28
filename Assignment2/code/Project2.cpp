#include "mainwindow.h"
#include "math.h"
#include "ui_mainwindow.h"
#include <QtGui>
#include "Matrix.h"
#include "time.h"

/*******************************************************************************
    The following are helper routines with code already written.
    The routines you'll need to write for the assignment are below.
*******************************************************************************/

/*******************************************************************************
Draw detected Harris corners
    interestPts - interest points
    numInterestsPts - number of interest points
    imageDisplay - image used for drawing

    Draws a red cross on top of detected corners
*******************************************************************************/
void MainWindow::DrawInterestPoints(CIntPt *interestPts, int numInterestsPts, QImage &imageDisplay)
{
   int i;
   int r, c, rd, cd;
   int w = imageDisplay.width();
   int h = imageDisplay.height();

   for(i=0;i<numInterestsPts;i++)
   {
       c = (int) interestPts[i].m_X;
       r = (int) interestPts[i].m_Y;

       for(rd=-2;rd<=2;rd++)
           if(r+rd >= 0 && r+rd < h && c >= 0 && c < w)
               imageDisplay.setPixel(c, r + rd, qRgb(255, 0, 0));

       for(cd=-2;cd<=2;cd++)
           if(r >= 0 && r < h && c + cd >= 0 && c + cd < w)
               imageDisplay.setPixel(c + cd, r, qRgb(255, 0, 0));
   }
}

/*******************************************************************************
Compute interest point descriptors
    image - input image
    interestPts - array of interest points
    numInterestsPts - number of interest points

    If the descriptor cannot be computed, i.e. it's too close to the boundary of
    the image, its descriptor length will be set to 0.

    I've implemented a very simple 8 dimensional descriptor.  Feel free to
    improve upon this.
*******************************************************************************/
void MainWindow::ComputeDescriptors(QImage image, CIntPt *interestPts, int numInterestsPts)
{
    int r, c, cd, rd, i, j;
    int w = image.width();
    int h = image.height();
    double *buffer = new double [w*h];
    QRgb pixel;

    // Descriptor parameters
    double sigma = 2.0;
    int rad = 4;

    // Computer descriptors from green channel
    for(r=0;r<h;r++)
       for(c=0;c<w;c++)
        {
            pixel = image.pixel(c, r);
            buffer[r*w + c] = (double) qGreen(pixel);
        }

    // Blur
    SeparableGaussianBlurImage(buffer, w, h, sigma);

    // Compute the desciptor from the difference between the point sampled at its center
    // and eight points sampled around it.
    for(i=0;i<numInterestsPts;i++)
    {
        int c = (int) interestPts[i].m_X;
        int r = (int) interestPts[i].m_Y;

        if(c >= rad && c < w - rad && r >= rad && r < h - rad)
        {
            double centerValue = buffer[(r)*w + c];
            int j = 0;

            for(rd=-1;rd<=1;rd++)
                for(cd=-1;cd<=1;cd++)
                    if(rd != 0 || cd != 0)
                {
                    interestPts[i].m_Desc[j] = buffer[(r + rd*rad)*w + c + cd*rad] - centerValue;
                    j++;
                }

            interestPts[i].m_DescSize = DESC_SIZE;
        }
        else
        {
            interestPts[i].m_DescSize = 0;
        }
    }

    delete [] buffer;
}

/*******************************************************************************
Draw matches between images
    matches - matching points
    numMatches - number of matching points
    image1Display - image to draw matches
    image2Display - image to draw matches

    Draws a green line between matches
*******************************************************************************/
void MainWindow::DrawMatches(CMatches *matches, int numMatches, QImage &image1Display, QImage &image2Display)
{
    int i;
    // Show matches on image
    QPainter painter;
    painter.begin(&image1Display);
    QColor green(0, 250, 0);
    QColor red(250, 0, 0);

    for(i=0;i<numMatches;i++)
    {
        painter.setPen(green);
        painter.drawLine((int) matches[i].m_X1, (int) matches[i].m_Y1, (int) matches[i].m_X2, (int) matches[i].m_Y2);
        painter.setPen(red);
        painter.drawEllipse((int) matches[i].m_X1-1, (int) matches[i].m_Y1-1, 3, 3);
    }

    QPainter painter2;
    painter2.begin(&image2Display);
    painter2.setPen(green);

    for(i=0;i<numMatches;i++)
    {
        painter2.setPen(green);
        painter2.drawLine((int) matches[i].m_X1, (int) matches[i].m_Y1, (int) matches[i].m_X2, (int) matches[i].m_Y2);
        painter2.setPen(red);
        painter2.drawEllipse((int) matches[i].m_X2-1, (int) matches[i].m_Y2-1, 3, 3);
    }

}


/*******************************************************************************
Given a set of matches computes the "best fitting" homography
    matches - matching points
    numMatches - number of matching points
    h - returned homography
    isForward - direction of the projection (true = image1 -> image2, false = image2 -> image1)
*******************************************************************************/
bool MainWindow::ComputeHomography(CMatches *matches, int numMatches, double h[3][3], bool isForward)
{
    int error;
    int nEq=numMatches*2;

    dmat M=newdmat(0,nEq,0,7,&error);
    dmat a=newdmat(0,7,0,0,&error);
    dmat b=newdmat(0,nEq,0,0,&error);

    double x0, y0, x1, y1;

    for (int i=0;i<nEq/2;i++)
    {
        if(isForward == false)
        {
            x0 = matches[i].m_X1;
            y0 = matches[i].m_Y1;
            x1 = matches[i].m_X2;
            y1 = matches[i].m_Y2;
        }
        else
        {
            x0 = matches[i].m_X2;
            y0 = matches[i].m_Y2;
            x1 = matches[i].m_X1;
            y1 = matches[i].m_Y1;
        }


        //Eq 1 for corrpoint
        M.el[i*2][0]=x1;
        M.el[i*2][1]=y1;
        M.el[i*2][2]=1;
        M.el[i*2][3]=0;
        M.el[i*2][4]=0;
        M.el[i*2][5]=0;
        M.el[i*2][6]=(x1*x0*-1);
        M.el[i*2][7]=(y1*x0*-1);

        b.el[i*2][0]=x0;
        //Eq 2 for corrpoint
        M.el[i*2+1][0]=0;
        M.el[i*2+1][1]=0;
        M.el[i*2+1][2]=0;
        M.el[i*2+1][3]=x1;
        M.el[i*2+1][4]=y1;
        M.el[i*2+1][5]=1;
        M.el[i*2+1][6]=(x1*y0*-1);
        M.el[i*2+1][7]=(y1*y0*-1);

        b.el[i*2+1][0]=y0;

    }
    int ret=solve_system (M,a,b);
    if (ret!=0)
    {
        freemat(M);
        freemat(a);
        freemat(b);

        return false;
    }
    else
    {
        h[0][0]= a.el[0][0];
        h[0][1]= a.el[1][0];
        h[0][2]= a.el[2][0];

        h[1][0]= a.el[3][0];
        h[1][1]= a.el[4][0];
        h[1][2]= a.el[5][0];

        h[2][0]= a.el[6][0];
        h[2][1]= a.el[7][0];
        h[2][2]= 1;
    }

    freemat(M);
    freemat(a);
    freemat(b);

    return true;
}


/*******************************************************************************
*******************************************************************************
*******************************************************************************

    The routines you need to implement are below

*******************************************************************************
*******************************************************************************
*******************************************************************************/

void ConvolveHelper(double *image, int width, int height, double *kernel, int kernelWidth, int kernelHeight)
{
    int kernelHalfHeight = (kernelHeight / 2);
    int kernelHalfWidth = (kernelWidth / 2);

    // Create and initialize our buffer
    int bufferHeight = (height + 2*kernelHalfHeight);
    int bufferWidth = (width + 2*kernelHalfWidth);
    double *buffer = new double[bufferWidth*bufferHeight];
    for (int by = 0; by < bufferHeight; by++)
    {
        for (int bx = 0; bx < bufferWidth; bx++)
        {
            int iy = (by - kernelHalfHeight);
            int ix = (bx - kernelHalfWidth);

            buffer[by*bufferWidth+bx] =
                    ((0 <= iy && iy < height && 0 <= ix && ix < width) ?
                         image[iy*width+ix] : 0.0);
        }
    }

    // Now, convolve the kernel over the image
    for (int iy = 0; iy < height; iy++)
    {
        for (int ix = 0; ix < width; ix++)
        {
            double result = 0.0;
            for (int ky = 0; ky < kernelHeight; ky++)
            {
                for (int kx = 0; kx < kernelWidth; kx++)
                {
                    double kernelWeight = kernel[ky*kernelWidth+kx];

                    // Translate to coordinates in buffer space
                    int by = iy + ky;
                    int bx = ix + kx;
                    result += kernelWeight*buffer[by*bufferWidth+bx];
                }
            }

            image[iy*width+ix] = result;
        }
    }

    // Clean up!
    delete[] buffer;
}

/*******************************************************************************
Blur a single channel floating point image with a Gaussian.
    image - input and output image
    w - image width
    h - image height
    sigma - standard deviation of Gaussian

    This code should be very similar to the code you wrote for assignment 1.
*******************************************************************************/
void MainWindow::SeparableGaussianBlurImage(double *image, int w, int h, double sigma)
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
    ConvolveHelper(image, w, h, kernel, kernelSize, 1);
    ConvolveHelper(image, w, h, kernel, 1, kernelSize);

    // Clean up!
    delete[] kernel;
}


/*******************************************************************************
Detect Harris corners.
    image - input image
    sigma - standard deviation of Gaussian used to blur corner detector
    thres - Threshold for detecting corners
    interestPts - returned interest points
    numInterestsPts - number of interest points returned
    imageDisplay - image returned to display (for debugging)
*******************************************************************************/
void MainWindow::HarrisCornerDetector(QImage image, double sigma, double thres, CIntPt **interestPts, int &numInterestsPts, QImage &imageDisplay)
{
    int r, c;
    int w = image.width();
    int h = image.height();
    double *buffer = new double [w*h];
    QRgb pixel;

    numInterestsPts = 0;

    // Compute the corner response using just the green channel
    for(r=0;r<h;r++)
       for(c=0;c<w;c++)
        {
            pixel = image.pixel(c, r);

            buffer[r*w + c] = (double) qGreen(pixel);
        }

    // Compute the x and y derivatives on the image
    double *derivX = new double[w*h];
    double *derivY = new double[w*h];
    for (int y = 0; y < h; y++)
    {
        for (int x = 0; x < w; x++)
        {
            derivX[y*w+x] = derivY[y*w+x] = buffer[y*w+x];
        }
    }
    double derivKernel[] = { -1, 0, 1 };
    ConvolveHelper(derivX, w, h, derivKernel, 1, 3);
    ConvolveHelper(derivY, w, h, derivKernel, 3, 1);

    // Compute derivX2, derivY2, and derivXY, then apply gaussian blur
    double *derivX2 = new double[w*h];
    double *derivY2 = new double[w*h];
    double *derivXY = new double[w*h];
    for (int y = 0; y < h; y++)
    {
        for (int x = 0; x < w; x++)
        {
            double dX = derivX[y*w+x];
            double dY = derivY[y*w+x];

            derivX2[y*w+x] = pow(dX, 2);
            derivY2[y*w+x] = pow(dY, 2);
            derivXY[y*w+x] = (dX * dY);
        }
    }
    SeparableGaussianBlurImage(derivX2, w, h, sigma);
    SeparableGaussianBlurImage(derivY2, w, h, sigma);
    SeparableGaussianBlurImage(derivXY, w, h, sigma);

    // Compute the Harris response
    double *harrisResponse = new double[w*h];
    for (int y = 0; y < h; y++)
    {
        for (int x = 0; x < w; x++)
        {
            double dX2 = derivX2[y*w+x];
            double dY2 = derivY2[y*w+x];
            double dXY = derivXY[y*w+x];

            // I'm just performing the math directly without constructing
            // the covariance matrix, H.  It would look like this:
            // [  dX2  dXY
            //    dXY  dY2  ]

            // Harris response: R = determinant(H) / trace(H)
            double determinant = (dX2 * dY2) - (dXY * dXY);
            double trace = (dX2 + dY2);
            harrisResponse[y*w+x] = (determinant / trace);
        }
    }

    // Find the peaks in the Harris response, and store them
    int numInterestsPtsTemp = 0;
    for (int y = 0; y < h; y++)
    {
        for (int x = 0; x < w; x++)
        {
            // First, check if we need to expand our temp interest point array
            if (numInterestsPts == numInterestsPtsTemp)
            {
                int newSize = numInterestsPtsTemp + 10;
                CIntPt *temp = new CIntPt[newSize];
                if (numInterestsPts > 0)
                {
                    for (int i = 0; i < numInterestsPtsTemp; i++)
                    {
                        temp[i].m_X = (*interestPts)[i].m_X;
                        temp[i].m_Y = (*interestPts)[i].m_Y;
                    }
                    delete[] (*interestPts);
                }
                *interestPts = temp;
                numInterestsPtsTemp = newSize;
            }

            // Peak response determination: if (x,y) is greater than its eight
            // neighbors, it's a peak response!
            bool fIsPeak = (harrisResponse[y*w+x] > thres);
            for (int j = -1; j <= 1 && fIsPeak; j++)
            {
                for (int i = -1; i <= 1 && fIsPeak; i++)
                {
                    int cy = y + j;
                    int cx = x + i;

                    // If we're in bounds (and not ourself), check to make sure we're a max!
                    if (0 <= cy && cy < h && 0 <= cx && cx < w && (cx != x || cy != y))
                    {
                        if (harrisResponse[y*w+x] <= harrisResponse[cy*w+cx])
                        {
                            fIsPeak = false;
                        }
                    }
                }
            }

            // If we're a peak, add to our interest point array!
            if (fIsPeak)
            {
                (*interestPts)[numInterestsPts].m_X = x;
                (*interestPts)[numInterestsPts].m_Y = y;
                numInterestsPts++;
            }
        }
    }

    // Now that we've found all the interest points we're going to find, shrink
    // the array of interest points if needed
    if (numInterestsPts != numInterestsPtsTemp)
    {
        CIntPt *temp = new CIntPt[numInterestsPts];
        for (int i = 0; i < numInterestsPts; i++)
        {
            temp[i].m_X = (*interestPts)[i].m_X;
            temp[i].m_Y = (*interestPts)[i].m_Y;
        }
        delete[] (*interestPts);
        *interestPts = temp;
    }

    // All done!  Draw the interest points, or the raw Harris response if
    // outputting for grading purposes
    bool fShowRawHarris = false;
    if (!fShowRawHarris)
    {
        DrawInterestPoints(*interestPts, numInterestsPts, imageDisplay);
    }
    else
    {
        // Find the maximum Harris response
        double maxResponse = -1;
        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
            {
                if (harrisResponse[y*w+x] > maxResponse)
                {
                    maxResponse = harrisResponse[y*w+x];
                }
            }
        }

        // Now output!
        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
            {
                int val = static_cast<int>(floor((harrisResponse[y*w+x]/maxResponse)*255+0.5));
                imageDisplay.setPixel(x, y, qRgb(val, val, val));
            }
        }
    }

    delete [] buffer;
    delete[] derivX;
    delete[] derivY;
    delete[] derivX2;
    delete[] derivY2;
    delete[] derivXY;
    delete[] harrisResponse;
}


/*******************************************************************************
Find matching interest points between images.
    image1 - first input image
    interestPts1 - interest points corresponding to image 1
    numInterestsPts1 - number of interest points in image 1
    image2 - second input image
    interestPts2 - interest points corresponding to image 2
    numInterestsPts2 - number of interest points in image 2
    matches - set of matching points to be returned
    numMatches - number of matching points returned
    image1Display - image used to display matches
    image2Display - image used to display matches
*******************************************************************************/
void MainWindow::MatchInterestPoints(QImage image1, CIntPt *interestPts1, int numInterestsPts1,
                             QImage image2, CIntPt *interestPts2, int numInterestsPts2,
                             CMatches **matches, int &numMatches, QImage &image1Display, QImage &image2Display)
{
    numMatches = 0;

    // Compute the descriptors for each interest point.
    // You can access the descriptor for each interest point using interestPts1[i].m_Desc[j].
    // If interestPts1[i].m_DescSize = 0, it was not able to compute a descriptor for that point
    ComputeDescriptors(image1, interestPts1, numInterestsPts1);
    ComputeDescriptors(image2, interestPts2, numInterestsPts2);

    // Find the best matches!
    int numMatchesTemp = 0;
    for (int img1 = 0; img1 < numInterestsPts1; img1++)
    {
        // Make sure this interest point has a valid descriptor
        if (interestPts1[img1].m_DescSize == 0)
        {
            continue;
        }

        // Next, check if we need to expand our matches array
        if (numMatches == numMatchesTemp)
        {
            int newSize = numMatchesTemp + 10;
            CMatches *temp = new CMatches[newSize];
            if (numMatches > 0)
            {
                for (int i = 0; i < numMatchesTemp; i++)
                {
                    temp[i] = (*matches)[i];
                }
                delete[] (*matches);
            }
            *matches = temp;
            numMatchesTemp = newSize;
        }

        int img2Closest = -1;
        double img2ClosestDistance = 0;

        for (int img2 = 0; img2 < numInterestsPts2; img2++)
        {
            // Make sure this interest point has a valid descriptor
            if (interestPts2[img2].m_DescSize == 0)
            {
                continue;
            }

            // Now, compare the two descriptors to see if they are good matches
            double distance = sqrt(
                    pow(interestPts1[img1].m_Desc[0]-interestPts2[img2].m_Desc[0], 2) +
                    pow(interestPts1[img1].m_Desc[1]-interestPts2[img2].m_Desc[1], 2) +
                    pow(interestPts1[img1].m_Desc[2]-interestPts2[img2].m_Desc[2], 2) +
                    pow(interestPts1[img1].m_Desc[3]-interestPts2[img2].m_Desc[3], 2) +
                    pow(interestPts1[img1].m_Desc[4]-interestPts2[img2].m_Desc[4], 2) +
                    pow(interestPts1[img1].m_Desc[5]-interestPts2[img2].m_Desc[5], 2) +
                    pow(interestPts1[img1].m_Desc[6]-interestPts2[img2].m_Desc[6], 2) +
                    pow(interestPts1[img1].m_Desc[7]-interestPts2[img2].m_Desc[7], 2)
                );
            if (img2Closest == -1 || distance < img2ClosestDistance)
            {
                img2Closest = img2;
                img2ClosestDistance = distance;
            }
        }

        // If we found a good match, add it to the array!
        if (img2Closest != -1)
        {
            (*matches)[numMatches].m_X1 = interestPts1[img1].m_X;
            (*matches)[numMatches].m_Y1 = interestPts1[img1].m_Y;
            (*matches)[numMatches].m_X2 = interestPts2[img2Closest].m_X;
            (*matches)[numMatches].m_Y2 = interestPts2[img2Closest].m_Y;
            numMatches++;
        }
    }

    // Now that we've found all the matches we're going to find, shrink the
    // array of matches if needed
    if (numMatches != numMatchesTemp)
    {
        CMatches *temp = new CMatches[numMatches];
        for (int i = 0; i < numMatches; i++)
        {
            temp[i] = (*matches)[i];
        }
        delete[] (*matches);
        *matches = temp;
    }

    // Draw the matches
    DrawMatches(*matches, numMatches, image1Display, image2Display);
}

/*******************************************************************************
Project a point (x1, y1) using the homography transformation h
    (x1, y1) - input point
    (x2, y2) - returned point
    h - input homography used to project point
*******************************************************************************/
void MainWindow::Project(double x1, double y1, double &x2, double &y2, double h[3][3])
{
    double u = h[0][0]*x1 + h[0][1]*y1 + h[0][2]*1;
    double v = h[1][0]*x1 + h[1][1]*y1 + h[1][2]*1;
    double w = h[2][0]*x1 + h[2][1]*y1 + h[2][2]*1;

    x2 = (u / w);
    y2 = (v / w);
}

/*******************************************************************************
Count the number of inliers given a homography.  This is a helper function for RANSAC.
    h - input homography used to project points (image1 -> image2
    matches - array of matching points
    numMatches - number of matchs in the array
    inlierThreshold - maximum distance between points that are considered to be inliers

    Returns the total number of inliers.
*******************************************************************************/
int MainWindow::ComputeInlierCount(double h[3][3], CMatches *matches, int numMatches, double inlierThreshold)
{
    int numInliers = 0;

    for (int i = 0; i < numMatches; i++)
    {
        double x2Projected;
        double y2Projected;
        Project(matches[i].m_X1, matches[i].m_Y1, x2Projected, y2Projected, h);

        double distance = sqrt(pow(matches[i].m_X2-x2Projected, 2) + pow(matches[i].m_Y2-y2Projected, 2));
        if (distance < inlierThreshold)
        {
            numInliers++;
        }
    }

    return numInliers;
}


/*******************************************************************************
Compute homography transformation between images using RANSAC.
    matches - set of matching points between images
    numMatches - number of matching points
    numIterations - number of iterations to run RANSAC
    inlierThreshold - maximum distance between points that are considered to be inliers
    hom - returned homography transformation (image1 -> image2)
    homInv - returned inverse homography transformation (image2 -> image1)
    image1Display - image used to display matches
    image2Display - image used to display matches
*******************************************************************************/
void MainWindow::RANSAC(CMatches *matches, int numMatches, int numIterations, double inlierThreshold,
                        double hom[3][3], double homInv[3][3], QImage &image1Display, QImage &image2Display)
{
    srand(time(nullptr));

    int numInliersBest = -1;
    double homComputedBest[3][3];

    // Over numIterations, try to determine the best homography
    for (int i = 0; i < numIterations; i++)
    {
        // Randomly select four matches
        int matchIndexes[4];
        for (int j = 0; j < 4; j++)
        {
            bool fUnique;
            do
            {
                matchIndexes[j] = (rand() % numMatches);

                // Check to make sure we haven't already selected this match
                fUnique = true;
                for (int k = 0; k < j; k++)
                {
                    if (matchIndexes[j] == matchIndexes[k])
                    {
                        fUnique = false;
                    }
                }
            } while (!fUnique);
        }

        // Using our four matches, compute the homography
        CMatches tempMatches[4];
        tempMatches[0] = matches[matchIndexes[0]];
        tempMatches[1] = matches[matchIndexes[1]];
        tempMatches[2] = matches[matchIndexes[2]];
        tempMatches[3] = matches[matchIndexes[3]];
        double homComputed[3][3];
        ComputeHomography(tempMatches, 4, homComputed, true);

        // Compute inlier count on the new homography
        int numInliers = ComputeInlierCount(homComputed, matches, numMatches, inlierThreshold);

        // Save homography if it has our highest number of inliers
        if (numInliers > numInliersBest)
        {
            numInliersBest = numInliers;
            homComputedBest[0][0] = homComputed[0][0];
            homComputedBest[0][1] = homComputed[0][1];
            homComputedBest[0][2] = homComputed[0][2];
            homComputedBest[1][0] = homComputed[1][0];
            homComputedBest[1][1] = homComputed[1][1];
            homComputedBest[1][2] = homComputed[1][2];
            homComputedBest[2][0] = homComputed[2][0];
            homComputedBest[2][1] = homComputed[2][1];
            homComputedBest[2][2] = homComputed[2][2];
        }
    }

    // Now that we've found a good homography to use, try to optimize it further
    // by taking the matches that are inliers and compute the homography using
    // all of them
    CMatches *inliers = new CMatches[numInliersBest];
    int numInliers = 0;
    for (int i = 0; i < numMatches; i++)
    {
        double x2Projected;
        double y2Projected;
        Project(matches[i].m_X1, matches[i].m_Y1, x2Projected, y2Projected, homComputedBest);
        double distance = sqrt(pow(matches[i].m_X2-x2Projected, 2) + pow(matches[i].m_Y2-y2Projected, 2));
        if (distance < inlierThreshold)
        {
            inliers[numInliers] = matches[i];
            numInliers++;
        }
    }
    ComputeHomography(inliers, numInliers, hom, true);
    ComputeHomography(inliers, numInliers, homInv, false);

    // After you're done computing the inliers, display the corresponding matches.
    DrawMatches(inliers, numInliers, image1Display, image2Display);

}

/*******************************************************************************
Bilinearly interpolate image (helper function for Stitch)
    image - input image
    (x, y) - location to interpolate
    rgb - returned color values

    You can just copy code from previous assignment.
*******************************************************************************/
bool MainWindow::BilinearInterpolation(QImage *image, double x, double y, double rgb[3])
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

    // What is this for?
    return true;
}


/*******************************************************************************
Stitch together two images using the homography transformation
    image1 - first input image
    image2 - second input image
    hom - homography transformation (image1 -> image2)
    homInv - inverse homography transformation (image2 -> image1)
    stitchedImage - returned stitched image
*******************************************************************************/
void MainWindow::Stitch(QImage image1, QImage image2, double hom[3][3], double homInv[3][3], QImage &stitchedImage)
{
    // Width and height of stitchedImage
    int ws = 0;
    int hs = 0;

    // Add your code to compute ws and hs here.

    stitchedImage = QImage(ws, hs, QImage::Format_RGB32);
    stitchedImage.fill(qRgb(0,0,0));

    // Add you code to warp image1 and image2 to stitchedImage here.
}

