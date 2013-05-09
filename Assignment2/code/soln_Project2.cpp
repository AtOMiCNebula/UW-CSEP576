#include "mainwindow.h"
#include "math.h"
#include "ui_mainwindow.h"
#include <QtGui>
#include "Matrix.h"

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
    double *XX = new double [w*h];
    double *XY = new double [w*h];
    double *YY = new double [w*h];
    double *harris = new double [w*h];
    QRgb pixel;

    for(r=0;r<h;r++)
       for(c=0;c<w;c++)
        {
            pixel = image.pixel(c, r);

            buffer[r*w + c] = (double) qGreen(pixel);
            XX[r*w + c] = 0.0;
            YY[r*w + c] = 0.0;
            XY[r*w + c] = 0.0;
        }

    for(r=1;r<h-1;r++)
       for(c=1;c<w-1;c++)
        {
            double val0, val1, val2, val3;

            val0 = buffer[r*w + c - 1];
            val1 = buffer[r*w + c + 1];
            val2 = buffer[(r-1)*w + c];
            val3 = buffer[(r+1)*w + c];

            XX[r*w + c] = (val1 - val0)*(val1 - val0);
            XY[r*w + c] = (val1 - val0)*(val3 - val2);
            YY[r*w + c] = (val3 - val2)*(val3 - val2);
        }

    SeparableGaussianBlurImage(XX, w, h, sigma);
    SeparableGaussianBlurImage(XY, w, h, sigma);
    SeparableGaussianBlurImage(YY, w, h, sigma);

    for(r=1;r<h-1;r++)
       for(c=1;c<w-1;c++)
        {
            double val, a, b, d;

            a = XX[r*w + c];
            b = XY[r*w + c];
            d = YY[r*w + c];

            val = 0.0;
            if(a + d > 0.0)
                val = (a*d - b*b)/(a + d);

            harris[r*w + c] = val;

            val *= 0.1;
            val += 50.0;

            if(val > 255.0)
                val = 255.0;

            imageDisplay.setPixel(c, r, qRgb((int) val, (int) val, (int) val));
        }

    numInterestsPts = 0;
    for(r=1;r<h-1;r++)
       for(c=1;c<w-1;c++)
        {
            int rd, cd;
            bool found = false;
            double val = harris[r*w + c];

            if(val > thres)
            {
                for(rd=-1;rd<=1;rd++)
                    for(cd=-1;cd<=1;cd++)
                    {
                        if(val < harris[(r+rd)*w + c + cd])
                            found = true;
                    }

                if(!found)
                {
                    numInterestsPts++;
                    imageDisplay.setPixel(c, r, qRgb((int) 255, (int) 0, (int) 0));
                }
            }
        }

    *interestPts = new CIntPt [numInterestsPts];
    numInterestsPts = 0;
    for(r=1;r<h-1;r++)
       for(c=1;c<w-1;c++)
        {
            QRgb pixel = imageDisplay.pixel(c, r);

            if(qRed(pixel) == 255 && qGreen(pixel) == 0)
            {
                (*interestPts)[numInterestsPts].m_X = (double) c;
                (*interestPts)[numInterestsPts].m_Y = (double) r;
                numInterestsPts++;
            }
        }

    DrawInterestPoints(*interestPts, numInterestsPts, imageDisplay);

    delete [] buffer;
    delete [] XX;
    delete [] XY;
    delete [] YY;
    delete [] harris;

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
    int i, j, k;

    ComputeDescriptors(image1, interestPts1, numInterestsPts1);
    ComputeDescriptors(image2, interestPts2, numInterestsPts2);

    int descSize = DESC_SIZE;


    numMatches = numInterestsPts1;
    *matches = new CMatches [numMatches];
    numMatches = 0;

    for(i=0;i<numInterestsPts1;i++)
        if(interestPts1[i].m_DescSize > 0)
    {
        int minIdx = 0;
        double minDistance = 99999999999.0;

        for(j=0;j<numInterestsPts2;j++)
            if(interestPts2[j].m_DescSize > 0)
        {
            double distance = 0.0;

            for(k=0;k<descSize;k++)
            {
                double difference = interestPts1[i].m_Desc[k] - interestPts2[j].m_Desc[k];
                distance += difference*difference;
            }

            if(distance < minDistance)
            {
                minDistance = distance;
                minIdx = j;
            }
        }

        (*matches)[numMatches].m_X1 = interestPts1[i].m_X;
        (*matches)[numMatches].m_Y1 = interestPts1[i].m_Y;
        (*matches)[numMatches].m_X2 = interestPts2[minIdx].m_X;
        (*matches)[numMatches].m_Y2 = interestPts2[minIdx].m_Y;
        numMatches++;
    }

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
    double u = h[0][0]*x1 + h[0][1]*y1 + h[0][2];
    double v = h[1][0]*x1 + h[1][1]*y1 + h[1][2];
    double w = h[2][0]*x1 + h[2][1]*y1 + h[2][2];

    x2 = u / w;
    y2 = v / w;
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
    int i;
    int ct = 0;

    for(i=0;i<numMatches;i++)
    {
        double x1, y1;

        Project(matches[i].m_X1, matches[i].m_Y1, x1, y1, h);

        double distance = (x1 - matches[i].m_X2)*(x1 - matches[i].m_X2) +
                          (y1 - matches[i].m_Y2)*(y1 - matches[i].m_Y2);

        if(distance < inlierThreshold*inlierThreshold)
            ct++;

    }

    return ct;
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
    int i, j, k;
    int maxNumInliers = 0;
    double h[3][3];

    CMatches match4[4];

    for(i=0;i<numIterations;i++)
    {
        for(j=0;j<4;j++)
        {
            int idx = rand()%numMatches;

            match4[j].m_X1 = matches[idx].m_X1;
            match4[j].m_Y1 = matches[idx].m_Y1;
            match4[j].m_X2 = matches[idx].m_X2;
            match4[j].m_Y2 = matches[idx].m_Y2;
        }

        bool hr = ComputeHomography(match4, 4, h, true);

        if(hr == true)
        {
            int ct = ComputeInlierCount(h, matches, numMatches, inlierThreshold);

            if(ct > maxNumInliers)
            {
                maxNumInliers = ct;

                for(j=0;j<3;j++)
                    for(k=0;k<3;k++)
                    {
                        hom[j][k] = h[j][k];
                    }

            }

        }
    }

    CMatches *inliers = new CMatches[maxNumInliers];

    int ct = 0;

    for(i=0;i<numMatches;i++)
    {
        double x1, y1;

        Project(matches[i].m_X1, matches[i].m_Y1, x1, y1, hom);

        double distance = (x1 - matches[i].m_X2)*(x1 - matches[i].m_X2) +
                          (y1 - matches[i].m_Y2)*(y1 - matches[i].m_Y2);

        if(distance < inlierThreshold*inlierThreshold)
        {
            inliers[ct].m_X1 = matches[i].m_X1;
            inliers[ct].m_Y1 = matches[i].m_Y1;
            inliers[ct].m_X2 = matches[i].m_X2;
            inliers[ct].m_Y2 = matches[i].m_Y2;

            ct++;
        }

    }

    ComputeHomography(inliers, ct, hom, true);
    ComputeHomography(inliers, ct, homInv, false);

    DrawMatches(inliers, ct, image1Display, image2Display);

    delete [] inliers;

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
    else
        return false;

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
    int i, r, c;
    double corners[4][2];
    int w1 = image1.width();
    int h1 = image1.height();
    int w2 = image2.width();
    int h2 = image2.height();
    int cs, ce, rs, re;
    FILE *out = fopen("debugs.txt", "w");

    cs = rs = 0;
    ce = w1;
    re = h1;

    corners[0][0] = 0.0;
    corners[0][1] = 0.0;
    corners[1][0] = (double) w2;
    corners[1][1] = 0.0;
    corners[2][0] = 0.0;
    corners[2][1] = (double) h2;
    corners[3][0] = (double) w2;
    corners[3][1] = (double) h2;

    for(i=0;i<4;i++)
    {
        double x, y;

        Project(corners[i][0], corners[i][1], x, y, homInv);

        int r = floor(y + 0.5);
        int c = floor(x + 0.5);

        if(cs > c)
            cs = c;
        if(ce < c)
            ce = c;
        if(rs > r)
            rs = r;
        if(re < r)
            re = r;
    }

    int ws = ce - cs;
    int hs = re - rs;

    stitchedImage = QImage(ws, hs, QImage::Format_RGB32);
    stitchedImage.fill(qRgb(0,0,0));

    for(r=0;r<h1;r++)
        for(c=0;c<w1;c++)
        {
            QRgb pixel = image1.pixel(c, r);
            stitchedImage.setPixel(c - cs, r - rs, pixel);
        }

    for(r=0;r<hs;r++)
        for(c=0;c<ws;c++)
        {
            double x, y, rgb[3];
            Project((double) (c+cs), (double) (r+rs), x, y, hom);

            bool hr = BilinearInterpolation(&image2, x, y, rgb);
            QRgb pixel = stitchedImage.pixel(c, r);

            if(hr)
            {
                if(qRed(pixel) != 0 || qGreen(pixel) != 0 || qBlue(pixel) != 0)
                    stitchedImage.setPixel(c, r, qRgb(((int) rgb[0] + qRed(pixel))/2,
                                                      ((int) rgb[1] + qGreen(pixel))/2,
                                                      ((int) rgb[2] + qBlue(pixel))/2));
                else
                    stitchedImage.setPixel(c, r, qRgb((int) rgb[0],
                                                      (int) rgb[1],
                                                      (int) rgb[2]));

            }
        }

    fclose(out);
}

