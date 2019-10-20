#include "ncnn_centerface.h"


Centerface::Centerface()
{
}

Centerface::~Centerface()
{
}

int Centerface::init(std::string model_path)
{
	std::string param = model_path + "/centerface.param";
	std::string bin= model_path + "/centerface.bin";
	net.load_param(param.data());
	net.load_model(bin.data());
	return 0;
}

int Centerface::detect(ncnn::Mat & inblob, std::vector<FaceInfo>& faces, int resized_w, int resized_h, float scoreThresh, float nmsThresh)
{
	if (inblob.empty()) {
		 std::cout << "blob is empty ,please check!" << std::endl;
		 return -1;
	}

	image_h = inblob.h;
	image_w = inblob.w;

	scale_w = (float)image_w / (float)resized_w;
	scale_h = (float)image_h / (float)resized_h;

	ncnn::Mat in;

	//scale 
	dynamicScale(resized_w, resized_h);
	ncnn::resize_bilinear(inblob, in, d_w, d_h);

	ncnn::Extractor ex = net.create_extractor();
	ex.input("input.1", in);

	ncnn::Mat heatmap, scale, offset, landmarks;
	ex.extract("537", heatmap);
	ex.extract("538", scale);
	ex.extract("539", offset);
	ex.extract("540", landmarks);


	decode(heatmap, scale,offset, landmarks,faces, scoreThresh,nmsThresh);
	squareBox(faces);
	return 0;
}

void Centerface::nms(std::vector<FaceInfo>& input, std::vector<FaceInfo>& output, float nmsthreshold)
{
	std::sort(input.begin(), input.end(),
		[](const FaceInfo& a, const FaceInfo& b)
	{
		return a.score > b.score;
	});

	int box_num = input.size();

	std::vector<int> merged(box_num, 0);

	for (int i = 0; i < box_num; i++)
	{
		if (merged[i])
			continue;

		output.push_back(input[i]);

		float h0 = input[i].y2 - input[i].y1 + 1;
		float w0 = input[i].x2 - input[i].x1 + 1;

		float area0 = h0 * w0;


		for (int j = i + 1; j < box_num; j++)
		{
			if (merged[j])
				continue;

			float inner_x0 = input[i].x1 > input[j].x1 ? input[i].x1 : input[j].x1;//std::max(input[i].x1, input[j].x1);
			float inner_y0 = input[i].y1 > input[j].y1 ? input[i].y1 : input[j].y1;

			float inner_x1 = input[i].x2 < input[j].x2 ? input[i].x2 : input[j].x2;  //bug fixed ,sorry
			float inner_y1 = input[i].y2 < input[j].y2 ? input[i].y2 : input[j].y2;

			float inner_h = inner_y1 - inner_y0 + 1;
			float inner_w = inner_x1 - inner_x0 + 1;


			if (inner_h <= 0 || inner_w <= 0)
				continue;

			float inner_area = inner_h * inner_w;

			float h1 = input[j].y2 - input[j].y1 + 1;
			float w1 = input[j].x2 - input[j].x1 + 1;

			float area1 = h1 * w1;

			float score;

			score = inner_area / (area0 + area1 - inner_area);

			if (score > nmsthreshold)
				merged[j] = 1;
		}

	}
}

void Centerface::decode(ncnn::Mat & heatmap, ncnn::Mat & scale, ncnn::Mat & offset, ncnn::Mat & landmarks, std::vector<FaceInfo>& faces, float scoreThresh, float nmsThresh)
{
	int fea_h = heatmap.h;
	int fea_w = heatmap.w;
	int spacial_size = fea_w*fea_h;

	float *heatmap_ = (float*)(heatmap.data);

	float *scale0 = (float*)(scale.data);
	float *scale1 = scale0 + spacial_size;

	float *offset0 = (float*)(offset.data);
	float *offset1 = offset0 + spacial_size;

	std::vector<int> ids;
	genIds(heatmap_, fea_h, fea_w, scoreThresh, ids);

	std::vector<FaceInfo> faces_tmp;
	for (int i = 0; i < ids.size() / 2; i++) {
		int id_h = ids[2 * i];
		int id_w = ids[2 * i + 1];
		int index = id_h*fea_w + id_w;

		float s0 = std::exp(scale0[index]) * 4;
		float s1 = std::exp(scale1[index]) * 4;
		float o0 = offset0[index];
		float o1 = offset1[index];

		//std::cout << s0 << " " << s1 << " " << o0 << " " << o1 << std::endl;

		float x1 = (id_w + o1 + 0.5) * 4 - s1 / 2 > 0.f ? (id_w + o1 + 0.5) * 4 - s1 / 2 : 0;
		float y1 =(id_h + o0 + 0.5) * 4 - s0 / 2 > 0 ? (id_h + o0 + 0.5) * 4 - s0 / 2 : 0;
		float x2 = 0, y2 = 0;
		x1 = x1 < (float)d_w ? x1 : (float)d_w;
		y1 = y1 < (float)d_h ? y1 : (float)d_h;
		x2 =  x1 + s1 < (float)d_w ? x1 + s1 : (float)d_w;
		y2 = y1 + s0 < (float)d_h ? y1 + s0 : (float)d_h;

		//std::cout << x1 << " " << y1 << " " << x2 << " " << y2 << std::endl;

		FaceInfo facebox;
		facebox.x1 = x1;
		facebox.y1 = y1;
		facebox.x2 = x2;
		facebox.y2 = y2;
		facebox.score = heatmap_[index];


		float box_w = x2 - x1; //=s1?
		float box_h = y2 - y1; //=s0?

		//std::cout << facebox.x1 << " " << facebox.y1 << " " << facebox.x2 << " " << facebox.y2 << std::endl;
		for (int j = 0; j < 5; j++) {
			float *xmap = (float*)landmarks.data + (2 * j + 1)*spacial_size;
			float *ymap = (float*)landmarks.data + (2 * j)*spacial_size;
			facebox.landmarks[2*j] = x1 + xmap[index] * s1;//box_w;
			facebox.landmarks[2 * j+1] = y1 + ymap[index] *  s0; // box_h;
		}
		faces_tmp.push_back(facebox);
	}

	nms(faces_tmp, faces, nmsThresh);

	for (int k = 0; k < faces.size(); k++) {
		faces[k].x1 *= d_scale_w*scale_w;
		faces[k].y1 *= d_scale_h*scale_h;
		faces[k].x2 *= d_scale_w*scale_w;
		faces[k].y2 *= d_scale_h*scale_h;

		for (int kk = 0; kk < 5; kk++) {
			faces[k].landmarks[2*kk]*= d_scale_w*scale_w;
			faces[k].landmarks[2*kk+1] *= d_scale_h*scale_h;
		}
	}
}

void Centerface::dynamicScale(float in_w, float in_h)
{
	d_h = (int)(std::ceil(in_h / 32) * 32);
	d_w = (int)(std::ceil(in_w / 32) * 32);

	d_scale_h = in_h / d_h;
	d_scale_w = in_w / d_w;
}

void Centerface::squareBox(std::vector<FaceInfo>& faces)
{
	float w = 0, h = 0, maxSize = 0;
	float cenx, ceny;
	for (int i = 0; i < faces.size(); i++) {
		w = faces[i].x2 - faces[i].x1;
		h = faces[i].y2 - faces[i].y1;

		maxSize = w < h ? h : w;
		cenx = faces[i].x1 + w / 2;
		ceny = faces[i].y1 + h / 2;

		faces[i].x1 =cenx - maxSize / 2 > 0 ? cenx - maxSize / 2 : 0;
		faces[i].y1 =ceny - maxSize / 2 > 0 ? ceny - maxSize / 2 : 0;
		faces[i].x2 = cenx + maxSize / 2 > image_w - 1 ? image_w - 1 : cenx + maxSize / 2;
		faces[i].y2 =ceny + maxSize / 2 > image_h-1 ? image_h - 1 : ceny + maxSize / 2;
	}
}

void Centerface::genIds(float * heatmap, int h, int w, float thresh, std::vector<int>& ids)
{
	if (heatmap==NULL)
	{
		std::cout << "heatmap is nullptr,please check! " << std::endl;
		return;
	}

	for (int i = 0; i < h; i++) {
		for (int j = 0; j < w; j++) {
			if (heatmap[i*w + j] > thresh) {
				ids.push_back(i);
				ids.push_back(j);
			}
		}
	}
}
