#ifndef _FACE_CENTERFACE_H_
#define _FACE_CENTERFACE_H_

#include <vector>

#include "MNN/Interpreter.hpp"
#include "MNN/ImageProcess.hpp"
#include "MNN/MNNDefine.h"
#include "MNN/Tensor.hpp"

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"

struct FaceInfo {
	cv::Rect face_;
	float score_;
	float keypoints_[10];
};

class Centerface {
public:
	Centerface();
	~Centerface();
	int Init(const char* model_path);
	int Detect(const cv::Mat& img_src, std::vector<FaceInfo>* faces);

private:
	uint8_t* GetImage(const cv::Mat& img_src) {
		uchar* data_ptr = new uchar[img_src.total() * 4];
		cv::Mat img_tmp(img_src.size(), CV_8UC4, data_ptr);
		cv::cvtColor(img_src, img_tmp, cv::COLOR_BGR2RGBA, 4);
		return (uint8_t*)img_tmp.data;
	}

	float InterRectArea(const cv::Rect& a, const cv::Rect& b) {
		cv::Point left_top = cv::Point(MAX(a.x, b.x), MAX(a.y, b.y));
		cv::Point right_bottom = cv::Point(MIN(a.br().x, b.br().x), MIN(a.br().y, b.br().y));
		cv::Point diff = right_bottom - left_top;
		return (MAX(diff.x + 1, 0) * MAX(diff.y + 1, 0));
	}

	int ComputeIOU(const cv::Rect& rect1,
		const cv::Rect& rect2, float* iou,
		const std::string& type = "UNION") {

		float inter_area = InterRectArea(rect1, rect2);
		if (type == "UNION") {
			*iou = inter_area / (rect1.area() + rect2.area() - inter_area);
		}
		else {
			*iou = inter_area / MIN(rect1.area(), rect2.area());
		}

		return 0;
	}


	int NMS(const std::vector<FaceInfo>& faces,
		std::vector<FaceInfo>* result, const float& threshold,
		const std::string& type = "UNION") {
		result->clear();
		if (faces.size() == 0)
			return -1;

		std::vector<size_t> idx(faces.size());

		for (unsigned i = 0; i < idx.size(); i++) {
			idx[i] = i;
		}

		while (idx.size() > 0) {
			int good_idx = idx[0];
			result->push_back(faces[good_idx]);
			std::vector<size_t> tmp = idx;
			idx.clear();
			for (unsigned i = 1; i < tmp.size(); i++) {
				int tmp_i = tmp[i];
				float iou = 0.0f;
				ComputeIOU(faces[good_idx].face_, faces[tmp_i].face_, &iou, type);
				if (iou <= threshold)
					idx.push_back(tmp_i);
			}
		}
	}

private:
	bool initialized_;
	std::shared_ptr<MNN::CV::ImageProcess> pretreat_;
	std::shared_ptr<MNN::Interpreter> centerface_interpreter_;
	MNN::Session* centerface_sess_ = nullptr;
	MNN::Tensor* input_tensor_ = nullptr;

	const float meanVals_[3] = { 0.0f, 0.0f, 0.0f };
	const float normVals_[3] = { 1.0f, 1.0f, 1.0f };
	const float scoreThreshold_ = 0.5f;
	const float nmsThreshold_ = 0.5f;

};


#endif  // !_FACE_CENTERFACE_H_
