#include "connectedComponents.h"

int connectedComponents(cv::Mat input,cv::Mat &output)
{
	int h= input.rows;
	int w = input.cols;

	int conneted = 8;

	std::vector<std::vector<cv::Point> > contours;

	std::vector<cv::Vec4i> hierarchy;


	cv::findContours( input, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE );


	cv::Mat dst = cv::Mat::zeros(input.size(), CV_8UC1);

    
	int label = 1;
    if( !contours.empty() && !hierarchy.empty() )
	{
        // iterate through all the top-level contours,

        // draw each connected component with its own random color

        int idx = 0;
		
		
        for( ; idx >= 0; idx = hierarchy[idx][0] )
		{
			
			//cv::Scalar color( (rand()&255), (rand()&255), (rand()&255) );
			cv::drawContours( dst, contours, idx, label, CV_FILLED, conneted, hierarchy );
			label++;
		}
	}
	
	dst.convertTo(output,CV_32SC1);

	return label;
}

int connectedComponents(cv::Mat input,cv::Mat &output,int conneted)
{
	int h= input.rows;
	int w = input.cols;


	std::vector<std::vector<cv::Point> > contours;

	std::vector<cv::Vec4i> hierarchy;


	cv::findContours( input, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE );


	cv::Mat dst = cv::Mat::zeros(input.size(), CV_8UC1);

    
	int label = 1;
    if( !contours.empty() && !hierarchy.empty() )
	{
        // iterate through all the top-level contours,

        // draw each connected component with its own random color

        int idx = 0;
		
		
        for( ; idx >= 0; idx = hierarchy[idx][0] )
		{
			
			//cv::Scalar color( (rand()&255), (rand()&255), (rand()&255) );
			cv::drawContours( dst, contours, idx, label, CV_FILLED, conneted, hierarchy );
			label++;
		}
	}
	
	dst.convertTo(output,CV_32SC1);

	return label;
}
