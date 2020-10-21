/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  Implementation of the Gaussian mixture model background subtraction from:
//
//  “BMOG: boosted Gaussian Mixture Model with controlled complexity for background subtraction”
//  I. Martins, P. Carvalho, L. Corte-Real, and J. L. Alba-Castro
//  Pattern Analysis and Appl., April, 2018
//  http://link.springer.com/article/10.1007/s10044-018-0699-y
//  A full-text view-only version of the paper is available at: http://rdcu.be/KtPr
//
//  Adapted from OpenCV-3.0.0 MOG2 implementation
//  Isabel Martins, 2017
//
//  Original BSD license:
M*/
/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#ifndef __OPENCV_BMOG_HPP__
#define __OPENCV_BMOG_HPP__


#include <iostream>


namespace cv
{
	/** @brief BMOG Background/Foreground Segmentation Algorithm.

	  The class implements the Gaussian mixture model background subtraction described in
	//  “BMOG: boosted Gaussian Mixture Model with controlled complexity for background subtraction”
	//  I. Martins, P. Carvalho, L. Corte-Real, and J. L. Alba-Castro
	//  Pattern Analysis and Appl., April, 2018
	//  http://link.springer.com/article/10.1007/s10044-018-0699-y
	*/

	class CV_EXPORTS_W BackgroundSubtractorBMOG : public BackgroundSubtractor
	{
		public:

			/** @brief Returns the number of last frames that affect the background model
			*/
			CV_WRAP virtual int getHistory() const = 0;
			/** @brief Sets the number of last frames that affect the background model
			*/
			CV_WRAP virtual void setHistory(int history) = 0;

			/** @brief Returns the number of gaussian components in the background model
			*/
			CV_WRAP virtual int getNMixtures() const = 0;
			/** @brief Sets the number of gaussian components in the background model.

			  The model needs to be reinitalized to reserve memory.
			  */
			CV_WRAP virtual void setNMixtures(int nmixtures) = 0;//needs reinitialization!

			/** @brief Returns the "background ratio" parameter of the algorithm

			  If a foreground pixel keeps semi-constant value for about backgroundRatio\*history frames, it's
			  considered background and added to the model as a center of a new component.
			  */
			CV_WRAP virtual double getBackgroundRatio() const = 0;
			/** @brief Sets the "background ratio" parameter of the algorithm
			*/
			CV_WRAP virtual void setBackgroundRatio(double ratio) = 0;

			/** @brief Returns the variance threshold for the pixel-model match

			  The main threshold on each color component difference to decide if the sample is well described by
			  the background model or not.
			  */
			CV_WRAP virtual double getVarThreshold_L() const = 0;
			CV_WRAP virtual double getVarThreshold_a() const = 0;
			CV_WRAP virtual double getVarThreshold_b() const = 0;
			/** @brief Sets the variance threshold for the pixel-model match
			*/
			CV_WRAP virtual void setVarThreshold_L(double varThreshold) = 0;
			CV_WRAP virtual void setVarThreshold_a(double varThreshold) = 0;
			CV_WRAP virtual void setVarThreshold_b(double varThreshold) = 0;

			/** @brief Returns the variance threshold for the pixel-model match used for new mixture component generation

			  Threshold for each color component difference that helps decide when a sample is close to the
			  existing components. If a pixel is not close to any component, it
			  is considered foreground or added as a new component. 3 sigma =\> Tg=3\*3=9 is default. A smaller Tg
			  value generates more components. A higher Tg value may result in a small number of components but
			  they can grow too large.
			  */
			CV_WRAP virtual double getVarThresholdGen() const = 0;
			/** @brief Sets the variance threshold for the pixel-model match used for new mixture component generation
			*/
			CV_WRAP virtual void setVarThresholdGen(double varThresholdGen) = 0;

			/** @brief Returns the initial variance of each gaussian component
			*/
			CV_WRAP virtual double getVarInit() const = 0;
			/** @brief Sets the initial variance of each gaussian component
			*/
			CV_WRAP virtual void setVarInit(double varInit) = 0;

			CV_WRAP virtual double getVarMin() const = 0;
			CV_WRAP virtual void setVarMin(double varMin) = 0;

			CV_WRAP virtual double getVarMax() const = 0;
			CV_WRAP virtual void setVarMax(double varMax) = 0;

			/** @brief Returns the complexity reduction threshold

			  This parameter defines the number of samples needed to accept to prove the component exists. CT=0.05
			  is a default value for all the samples.
			  */
			CV_WRAP virtual double getComplexityReductionThreshold() const = 0;
			/** @brief Sets the complexity reduction threshold
			*/
			CV_WRAP virtual void setComplexityReductionThreshold(double ct) = 0;

			/** @brief Returns the size of the median filter used in PostProcessing
			*/
			CV_WRAP virtual int getPostProcessingSize() const = 0;
			/** @brief Sets the the size of the median filter used in PostProcessing
			*/
			CV_WRAP virtual void setPostProcessingSize(int value) = 0;

			/** @brief Returns the number of initial frames using original MOG2 algorithm
			  before applying BMOG adaptation rules. This allows building a reliable model before using adaptation rules that rely on previous decisions.
			  */
			CV_WRAP virtual int getTransientFrames() const = 0;
			/** @brief Sets the number of initial frames using MOG2
			*/
			CV_WRAP virtual void setTransientFrames(int value) = 0;

			/** @brief Returns the color space selected: 0->L*a*b*  1->YUV  2->YCbCr
			*/
			CV_WRAP virtual int getColorSpace() const = 0;
			/** @brief Sets the color space selected: 0->L*a*b*  1->YUV  2->YCbCr
			*/
			CV_WRAP virtual void setColorSpace(int value) = 0;
	};


	/** @brief Creates BMOG Background Subtractor

	  @param history Length of the history.
	  @param varThresholdL and varThresholdab Thresholds on the squared Mahalanobis distance between the pixel and the model
	  to decide whether a pixel is well described by the background model. These parameters do not
	  affect the background update.
	  */
	CV_EXPORTS_W Ptr<BackgroundSubtractorBMOG>
		createBackgroundSubtractorBMOG(int history=100, double varThresholdL=35, double varThresholdab=12);

	//@} //video_motion

	Ptr<BackgroundSubtractorBMOG> bgBMOG;

	extern "C" void getfg(int rows, int cols, unsigned char* imgData,
        unsigned char *fgD) {
		cv::Mat img(rows, cols, CV_8UC3, (void *) imgData);
		cv::Mat fg(rows, cols, CV_8UC1, fgD);
		bgBMOG->apply(img, fg);
	}

	extern "C" void getbg(int rows, int cols, unsigned char* bgD) {
		cv::Mat bg(rows, cols, CV_8UC3, bgD);
		bgBMOG->getBackgroundImage(bg);
	}

	extern "C" void init_bgs(int nmixtures, int threshold_L, int threshold_a, int threshold_b, double backgroundRatio, int postProcessingSize) {
        bgBMOG = createBackgroundSubtractorBMOG();

        // defaults
        // bgBMOG->setNMixtures( 5 );
		// bgBMOG->setVarThreshold_L( 35 );
		// bgBMOG->setVarThreshold_a( 12 );
		// bgBMOG->setVarThreshold_b( 12 );
		// bgBMOG->setPostProcessingSize( 9 );
        // bgBMOG->setBackgroundRatio( 1.0 );
        // ----
        //bgBMOG->setTransientFrames( 50 );
		//bgBMOG->setVarThresholdGen( 8.0 );
		//bgBMOG->setVarInit( 11.0 );
		//bgBMOG->setHistory( 100 );
		//bgBMOG->setComplexityReductionThreshold( 0.05 );

        // set new BMOG parameters
		bgBMOG->setNMixtures(nmixtures);
		bgBMOG->setVarThreshold_L(threshold_L);
		bgBMOG->setVarThreshold_a(threshold_a);
		bgBMOG->setVarThreshold_b(threshold_b);
		bgBMOG->setBackgroundRatio(backgroundRatio);
		bgBMOG->setPostProcessingSize(postProcessingSize);
	}

} // cv

#endif
