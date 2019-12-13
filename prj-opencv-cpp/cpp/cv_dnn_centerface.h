#pragma once
#include<string>
#include<vector>
#include<algorithm>
#include <numeric>
#include<math.h>
#include<opencv2/opencv.hpp>


typedef struct FaceInfo {
	float x1;
	float y1;
	float x2;
	float y2;
	float score;
	float landmarks[10];
};


class Centerface {
public:
	Centerface(std::string model_path,int width,int height);
	~Centerface();

	void detect(cv::Mat &image, std::vector<FaceInfo>&faces, float scoreThresh = 0.5,float nmsThresh=0.3);

private:
	void nms(std::vector<FaceInfo>& input, std::vector<FaceInfo>& output,float nmsthreshold=0.3);
	void decode(cv::Mat &heatmap,cv::Mat &scale,cv::Mat &offset,cv::Mat &landmarks, std::vector<FaceInfo>&faces,float scoreThresh,float nmsThresh);
	void dynamic_scale(float in_w,float in_h);
	std::vector<int> getIds(float *heatmap,int h,int w,float thresh);
	void squareBox(std::vector<FaceInfo> &faces);
private:
	int d_h;
	int d_w;
	float d_scale_h;
	float d_scale_w;

	float scale_w ;
	float scale_h ;

	int image_h;
	int image_w;
	
	cv::dnn::Net net;
};


