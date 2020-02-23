#include <iostream>
#include <opencv2/opencv.hpp>
#include "centerface.h"

int main(int argc, char** argv) {
	if (argc !=3) {
		std::cout << " .exe mode_path image_file" << std::endl;
		return -1;
	}

	const char* model_path = argv[1];
	const char* image_file = argv[2];

	Centerface* centerface = new Centerface();

	centerface->Init(model_path);

	cv::Mat img_src = cv::imread(image_file);

	std::vector<FaceInfo> faces;
	centerface->Detect(img_src, &faces);

	int num_faces = static_cast<float>(faces.size());
	for (int i = 0; i < num_faces; ++i) {
		cv::rectangle(img_src, faces[i].face_, cv::Scalar(255, 0, 255), 2);
		for (int j = 0; j < 5; ++j) {
			cv::Point curr_pt = cv::Point(faces[i].keypoints_[2 * j], faces[i].keypoints_[2 * j + 1]);
			cv::circle(img_src, curr_pt, 2, cv::Scalar(255, 0, 255), 2);
		}
	}

	cv::imshow("result", img_src);
	cv::waitKey(0);

    delete centerface;
	return 0;
}