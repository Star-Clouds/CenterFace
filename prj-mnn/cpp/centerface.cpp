#include "centerface.h"

#include <algorithm>
#include <iostream>
#include <string>

#include "opencv2/imgproc.hpp"

Centerface::Centerface() {
	initialized_ = false;
}

Centerface::~Centerface() {
	centerface_interpreter_->releaseModel();
	centerface_interpreter_->releaseSession(centerface_sess_);
}

int Centerface::Init(const char * model_path) {
	std::cout << "start init." << std::endl;
	std::string model_file = std::string(model_path) + "/centerface.mnn";
	centerface_interpreter_ = std::unique_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(model_file.c_str()));
	if (nullptr == centerface_interpreter_) {
		std::cout << "load centerface failed." << std::endl;
		return 10000;
	}

	MNN::ScheduleConfig schedule_config;
	schedule_config.type = MNN_FORWARD_CPU;
	schedule_config.numThread = 4;
	MNN::BackendConfig backend_config;
	backend_config.power = MNN::BackendConfig::Power_High;
	backend_config.precision = MNN::BackendConfig::Precision_High;
	schedule_config.backendConfig = &backend_config;

	// create session
	centerface_sess_ = centerface_interpreter_->createSession(schedule_config);
	input_tensor_ = centerface_interpreter_->getSessionInput(centerface_sess_, nullptr);

	MNN::CV::Matrix trans;
	trans.setScale(1.0f, 1.0f);
	MNN::CV::ImageProcess::Config img_config;
	img_config.filterType = MNN::CV::BICUBIC;
	::memcpy(img_config.mean, meanVals_, sizeof(meanVals_));
	::memcpy(img_config.normal, normVals_, sizeof(normVals_));
	img_config.sourceFormat = MNN::CV::RGBA;
	img_config.destFormat = MNN::CV::RGB;
	pretreat_ = std::shared_ptr<MNN::CV::ImageProcess>(MNN::CV::ImageProcess::create(img_config));
	pretreat_->setMatrix(trans);

	initialized_ = true;

	std::cout << "end init." << std::endl;
	return 0;
}

int Centerface::Detect(const cv::Mat & img_src, std::vector<FaceInfo>* faces) {
	std::cout << "start detect." << std::endl;
	faces->clear();
	if (!initialized_) {
		std::cout << "model uninitialized." << std::endl;
		return 10000;
	}
	if (img_src.empty()) {
		std::cout << "input empty." << std::endl;
		return 10001;
	}

	int width = img_src.cols;
	int height = img_src.rows;
	int width_resized = width / 32 * 32;
	int height_resized = height / 32 * 32;
	float scale_x = static_cast<float>(width) / width_resized;
	float scale_y = static_cast<float>(height) / height_resized;

	centerface_interpreter_->resizeTensor(input_tensor_, {1, 3, height_resized, width_resized});
	centerface_interpreter_->resizeSession(centerface_sess_);

	cv::Mat img_resized;
	cv::resize(img_src, img_resized, cv::Size(width_resized, height_resized));
	uint8_t* data_ptr = GetImage(img_resized);
	pretreat_->convert(data_ptr, width_resized, height_resized, 0, input_tensor_);

	// run session
	centerface_interpreter_->runSession(centerface_sess_);

	// get output
	MNN::Tensor* tensor_heatmap = centerface_interpreter_->getSessionOutput(centerface_sess_, "537");
	MNN::Tensor* tensor_scale = centerface_interpreter_->getSessionOutput(centerface_sess_, "538");
	MNN::Tensor* tensor_offset = centerface_interpreter_->getSessionOutput(centerface_sess_, "539");
	MNN::Tensor* tensor_landmark = centerface_interpreter_->getSessionOutput(centerface_sess_, "540");

	// copy to host
	MNN::Tensor heatmap_host(tensor_heatmap, tensor_heatmap->getDimensionType());
	MNN::Tensor scale_host(tensor_scale, tensor_scale->getDimensionType());
	MNN::Tensor offset_host(tensor_offset, tensor_offset->getDimensionType());
	MNN::Tensor landmark_host(tensor_landmark, tensor_landmark->getDimensionType());
	tensor_heatmap->copyToHostTensor(&heatmap_host);
	tensor_scale->copyToHostTensor(&scale_host);
	tensor_offset->copyToHostTensor(&offset_host);
	tensor_landmark->copyToHostTensor(&landmark_host);

	int output_width = heatmap_host.width();
	int output_height = heatmap_host.height();
	int channel_step = output_width * output_height;
	std::vector<FaceInfo> faces_tmp;
	for (int h = 0; h < output_height; ++h) {
		for (int w = 0; w < output_width; ++w) {
			int index = h * output_width + w;
			float score = heatmap_host.host<float>()[index];
			if (score < scoreThreshold_) {
				continue;
			}
			float s0 = 4 * exp(scale_host.host<float>()[index]);
			float s1 = 4 * exp(scale_host.host<float>()[index + channel_step]);
			float o0 = offset_host.host<float>()[index];
			float o1 = offset_host.host<float>()[index + channel_step];

			float ymin = MAX(0, 4 * (h + o0 + 0.5) - 0.5 * s0);
			float xmin = MAX(0, 4 * (w + o1 + 0.5) - 0.5 * s1);
			float ymax = MIN(ymin + s0, height_resized);
			float xmax = MIN(xmin + s1, width_resized);

			FaceInfo face_info;
			face_info.score_ = score;
			face_info.face_.x = scale_x * xmin;
			face_info.face_.y = scale_y * ymin;
			face_info.face_.width = scale_x * (xmax - xmin);
			face_info.face_.height = scale_y * (ymax - ymin);

			for (int num = 0; num < 5; ++num) {
				face_info.keypoints_[2 * num] = scale_x * (s1 * landmark_host.host<float>()[(2 * num + 1) * channel_step + index] + xmin);
				face_info.keypoints_[2 * num + 1] = scale_y * (s0 * landmark_host.host<float>()[(2 * num + 0) * channel_step + index] + ymin);
			}
			faces_tmp.push_back(face_info);
		}
	}
    std::sort(faces_tmp.begin(), faces_tmp.end(),
    [](const FaceInfo& a, const FaceInfo& b) {
        return a.score_ > b.score_;
    });
	NMS(faces_tmp, faces, nmsThreshold_);
	std::cout << "end detect." << std::endl;
	return 0;
}


