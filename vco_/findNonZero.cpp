#include "findNonZero.h"

void findNonZero(cv::Mat input,std::vector<cv::Point> &output)
{
	int h = input.rows;
	int w = input.cols;

	std::vector<cv::Point> idx;

	int cnt=0;
	for(int y=0; y < h; y++)
	for(int x=0; x < w; x++)
	{
		if(input.at<uchar>(y,x))
		{
			idx.push_back( cv::Point(x,y));
		
			cnt++;
		}
	}
	
	output = idx;
}
void findNonZero(cv::Mat input,cv::Mat &output)
{
	int h = input.rows;
	int w = input.cols;

	cv::Mat idx;

	int cnt=0;
	for(int y=0; y < h; y++)
	for(int x=0; x < w; x++)
	{
		if(input.at<uchar>(y,x))
		{
			if(!cnt)
			{
				idx = cv::Mat::zeros(1,1,CV_32SC2);

				idx.at<cv::Point>(0,0) = cv::Point(x,y);
			}
			else
			{
				idx.push_back( cv::Point(x,y));
			}

			cnt++;
		}
	}
	
	idx.copyTo(output);
	
}