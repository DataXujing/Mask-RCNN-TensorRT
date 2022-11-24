
// xujing
// 2022-11-24

// Mask RCNN TensorRT 实现，因官方实现进行了多层的封装且不能直接支持图像调用
// 基于识别逻辑重新实现了人可以看懂的代码！


#include <iostream>
#include <fstream>
#include <numeric>
#include <cstdio>
#include <stdio.h>
#include <math.h>
#include <string>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/dnn.hpp>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "cuda_runtime_api.h"
#include "NvOnnxParser.h"
#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "logging.h"
// max
#include <algorithm>
// MaskRCNN Parameter
#include "mrcnn_config.h"


using namespace sample;
using namespace std;
using namespace cv;

#define INPUT_SIZE  1024

float confThreshold = 0.5; // Confidence threshold
float maskThreshold = 0.3; // Mask threshold


//struct RawDetection
//{
//	float y1, x1, y2, x2, class_id, score;
//};

//struct Mask
//{
//	float raw[MaskRCNNConfig::MASK_POOL_SIZE * 2 * MaskRCNNConfig::MASK_POOL_SIZE * 2];
//};

struct Bbox {
	float x1;
	float y1;
	float x2;
	float y2;
};

struct BBoxInfo
{
	Bbox box;
	int label = -1;
	float prob = 0.0f;

	float mask [MaskRCNNConfig::MASK_POOL_SIZE * 2 * MaskRCNNConfig::MASK_POOL_SIZE * 2];
};


//前处理
// 0.RGB转BGR,1.等比例缩放（只缩小，不放大），2.bilinear interpolation resize 3. 上下左右填充0
//4.mold_image  （R-123.7),(G-116.8),(B-103.9)
//MEAN_PIXEL = np.array([123.7, 116.8, 103.9])
void preprocess(cv::Mat& img, float data[]) {

	cv::Mat rgb(img.rows, img.cols, CV_8UC3);
	cv::cvtColor(img, rgb, cv::COLOR_BGR2RGB);

	int w, h, x, y;
	float r_w = INPUT_SIZE / (img.cols*1.0);
	float r_h = INPUT_SIZE / (img.rows*1.0);

	if (r_h > r_w) {
		w = INPUT_SIZE;
		h = r_w * img.rows;
		x = 0;
		y = (INPUT_SIZE - h) / 2;
	}
	else {
		w = r_h * img.cols;
		h = INPUT_SIZE;
		x = (INPUT_SIZE - w) / 2;
		y = 0;
	}

	cv::Mat re(h, w, CV_8UC3);
	cv::resize(rgb, re, re.size(), 0, 0, cv::INTER_LINEAR);

	cv::Mat out(INPUT_SIZE, INPUT_SIZE, CV_8UC3, cv::Scalar(0, 0, 0));
	re.copyTo(out(cv::Rect(x, y, re.cols, re.rows)));


   //x-x_mean
	//re.convertTo(re, CV_32FC3, 1 / 1.0); // 转float 归一化
	std::vector<cv::Mat> rgbChannels(3);
	std::vector<float> dstdata;
	cv::split(out, rgbChannels);
	for (auto i = 0; i < rgbChannels.size(); i++) {
		std::vector<float> data_re = std::vector<float>(rgbChannels[i].reshape(1, 1));

		for (int j = 0; j < data_re.size(); j++) {
			if (i == 0) {
				dstdata.push_back((data_re[j] - 123.7));
			}
			else if (i == 1) {
				dstdata.push_back(data[j] - 116.8);
			}
			else {
				dstdata.push_back(data[j] -103.9);
			}
		}
	}

	std::copy(dstdata.begin(), dstdata.end(), data);
}

//后处理
void decodeOutput(std::vector<float> imginfo, float* detectionsHost, float* masksHost, std::vector<BBoxInfo> *pBInfo)
{
	int input_dim_h = MaskRCNNConfig::IMAGE_SHAPE.d[1], input_dim_w = MaskRCNNConfig::IMAGE_SHAPE.d[2];

	int image_height = imginfo[0];  //img的h，w
	int image_width = imginfo[1];
	// resize the DsImage with scale
	const int image_dim = std::max(image_height, image_width);
	int resizeH = (int)image_height * input_dim_h / (float)image_dim;
	int resizeW = (int)image_width * input_dim_w / (float)image_dim;
	// keep accurary from (float) to (int), then to float
	float window_x = (1.0f - (float)resizeW / input_dim_w) / 2.0f;
	float window_y = (1.0f - (float)resizeH / input_dim_h) / 2.0f;
	float window_width = (float)resizeW / input_dim_w;
	float window_height = (float)resizeH / input_dim_h;

	float final_ratio_x = (float)image_width / window_width;
	float final_ratio_y = (float)image_height / window_height;

	//std::vector<BBoxInfo> binfo;

	for (int det_id = 0; det_id < MaskRCNNConfig::DETECTION_MAX_INSTANCES; det_id++)
	{
		// 解析box y1, x1, y2, x2, class_id, score;
		int label = (int)detectionsHost[det_id * 6+4];
		if (label <= 0)
			continue;

		BBoxInfo det;
		det.label = label;
		det.prob = detectionsHost[det_id * 6 + 5];

		det.box.x1 = std::min(std::max((detectionsHost[det_id * 6 + 1] - window_x) * final_ratio_x, 0.0f), (float)image_width);
		det.box.y1 = std::min(std::max((detectionsHost[det_id * 6] - window_y) * final_ratio_y, 0.0f), (float)image_height);
		det.box.x2 = std::min(std::max((detectionsHost[det_id * 6 + 3] - window_x) * final_ratio_x, 0.0f), (float)image_width);
		det.box.y2 = std::min(std::max((detectionsHost[det_id * 6 + 2] - window_y) * final_ratio_y, 0.0f), (float)image_height);

		if (det.box.x2 <= det.box.x1 || det.box.y2 <= det.box.y1)
			continue;
		//解析mask
		for (int j = 0; j < 28 * 28 ; j++) {
			det.mask[j] = masksHost[det_id * 81 * 28 * 28 + label * 28 * 28 + j ];
			
		}

		pBInfo->push_back(det);
	}

	//return binfo;
}



//将28x28的mask转换到box大小
cv::Mat resizeMask(const BBoxInfo& box, const float mask_threshold)
{
	const int h = box.box.y2 - box.box.y1;
	const int w = box.box.x2 - box.box.x1;

	cv::Mat result(h, w, CV_32FC1, 0.0);
	if (!box.mask)
	{
		return result;
	}

	float new_mask[MaskRCNNConfig::MASK_POOL_SIZE * 2][MaskRCNNConfig::MASK_POOL_SIZE * 2];
	for (int i = 0; i < MaskRCNNConfig::MASK_POOL_SIZE * 2 ; i++) {
		for (int j = 0; j < MaskRCNNConfig::MASK_POOL_SIZE * 2; j++)
		if (box.mask[i*MaskRCNNConfig::MASK_POOL_SIZE * 2+j] > mask_threshold)
		{
			new_mask[i][j] = 255;
		}
		else {
			new_mask[i][j] = 0;
		}
	}

	cv::Mat raw_mask(MaskRCNNConfig::MASK_POOL_SIZE * 2, MaskRCNNConfig::MASK_POOL_SIZE * 2, CV_32FC1, new_mask);
	cv::resize(raw_mask, result, cv::Size(w, h), 0, 0, cv::INTER_LINEAR);   //数据格式又变为 CV_8UC1

	//cv::imwrite("./hahahah.jpg", result);
	return result;
}

// Draw the predicted bounding box, colorize and show the mask on the image
void drawBox(Mat& image, std::vector<BBoxInfo> bbinfos)
{

	for (int i = 0; i < bbinfos.size(); i++) {
		int x1 = bbinfos[i].box.x1;
		int y1 = bbinfos[i].box.y1;
		int x2 = bbinfos[i].box.x2;
		int y2 = bbinfos[i].box.y2;
		std::vector<int> color = { rand() % 256, rand() % 256, rand() % 256 };

		string label = MaskRCNNConfig::CLASS_NAMES[bbinfos[i].label] + format("%.2f", bbinfos[i].prob);

		//plot box
		//Draw a rectangle displaying the bounding box
		cv::rectangle(image, Point(x1, y1), Point(x2, y2), Scalar(color[0], color[1], color[2]), 3);

		//Display the label at the top of the bounding box
		int baseLine;
		Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
		y1 = max(y1, labelSize.height);
		rectangle(image, Point(x1, y1 - round(1.5*labelSize.height)), Point(x1 + round(1.5*labelSize.width), y1 + baseLine), Scalar(color[0], color[1], color[2]), FILLED);
		putText(image, label, Point(x1, y1), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 0, 0), 1);

		//plot mask
		cv::Mat mask = resizeMask(bbinfos[i], maskThreshold);

		image.convertTo(image, CV_32FC3);
		for (int i = 0; i < mask.rows; i++) {
			for (int j = 0; j < mask.cols; j++){

				mask.convertTo(mask, CV_32FC1, 1 / 1.0);
				float mask_val = mask.at<float>(i,j);
			
				if (mask_val >= 100) {
					//std::cout << mask_val << std::endl;

					int cur_y = y1 + i;
					int cur_x = x1 + j;

					if ((cur_x < image.size[1]) & (cur_y < image.size[0])) {
						image.at<Vec3f>(cur_y, cur_x)[0] = image.at<Vec3f>(cur_y, cur_x)[0] * 0.5 + color[0] * 0.5;
						image.at<Vec3f>(cur_y, cur_x)[1] = image.at<Vec3f>(cur_y, cur_x)[1] * 0.5 + color[1] * 0.5;
						image.at<Vec3f>(cur_y, cur_x)[2] = image.at<Vec3f>(cur_y, cur_x)[2] * 0.5 + color[2] * 0.5;
					}
				}
			
}
		}

	}

}




int main()
{
	Logger gLogger;
	//初始化插件，调用plugin必须初始化plugin respo
    nvinfer1:initLibNvInferPlugins(&gLogger, "");


	nvinfer1::IRuntime* engine_runtime = nvinfer1::createInferRuntime(gLogger);
	std::string engine_filepath = "./model/mask.plan";

	std::ifstream file;
	file.open(engine_filepath, std::ios::binary | std::ios::in);
	file.seekg(0, std::ios::end);
	int length = file.tellg();
	file.seekg(0, std::ios::beg);

	std::shared_ptr<char> data(new char[length], std::default_delete<char[]>());
	file.read(data.get(), length);
	file.close();

	//nvinfer1::ICudaEngine* engine_infer = engine_runtime->deserializeCudaEngine(cached_engine.data(), cached_engine.size(), nullptr);
	nvinfer1::ICudaEngine* engine_infer = engine_runtime->deserializeCudaEngine(data.get(), length, nullptr);
	nvinfer1::IExecutionContext* engine_context = engine_infer->createExecutionContext();

	int input_index = engine_infer->getBindingIndex("input_image"); //1x3x1024x1024
	//int input_index_1 = engine_infer->getBindingIndex("origin_input_resolution"); //w,h,w,h
	//std::string input_name = engine_infer->getBindingName(1);
	//std::cout << input_name << std::endl;
	int output_index_1 = engine_infer->getBindingIndex("mrcnn_detection");  //1
	int output_index_2 = engine_infer->getBindingIndex("mrcnn_mask/Sigmoid");   // 2



	std::cout << "输入的index: " << input_index << " 输出的mrcnn_detection-> " << output_index_1 << " mrcnn_mask/Sigmoid-> " << output_index_2  << std::endl;

	if (engine_context == nullptr)
	{
		std::cerr << "Failed to create TensorRT Execution Context." << std::endl;
	}

	// cached_engine->destroy();
	std::cout << "loaded trt model , do inference" << std::endl;


	//cv2读图片
	//cv::Mat image;
	//image = cv::imread("./test_3.jpg", 1);
	//int w = image.cols;
	//int h = image.rows;
	//float h_input_1[4] = { w, h, w, h  };


	//cv::Mat image;
	//image = cv::imread(fn[i], 1);

	cv::String pattern = "./test/*.jpg";
	std::vector<cv::String> fn;
	cv::glob(pattern, fn, false);
	std::vector<cv::Mat> images;
	size_t count = fn.size(); //number of png files in images folde

	std::cout << count << std::endl;

	float *h_input_0 = new float[INPUT_SIZE * INPUT_SIZE * 3];

	float *h_output_box = new float[100 * 6];   //1
	float *h_output_mask = new float[100 * 81 * 28 * 28];   //1


	for (size_t i = 0; i < count; i++)
	{
		cv::Mat image = cv::imread(fn[i]);
		cv::Mat image_origin = image.clone();

		std::cout << fn[i] << std::endl;


		float w = image.cols;
		float h = image.rows;

		memset(h_input_0, 0, INPUT_SIZE * INPUT_SIZE * 3 * sizeof(float));
		preprocess(image, h_input_0);

		void* buffers[3];
		cudaMalloc(&buffers[0], INPUT_SIZE * INPUT_SIZE * 3 * sizeof(float));  //<- input
		cudaMalloc(&buffers[1], 100 * 6 * sizeof(float));  //<- detection
		cudaMalloc(&buffers[2], 100 * 81 * 28 * 28 * sizeof(float)); //<- mask

		
		//cudaMemset(&buffers[0], 0, INPUT_SIZE * INPUT_SIZE * 3 * sizeof(float));
		//cudaMemset(&buffers[1], 0, 100 * 6 * sizeof(float));
		//cudaMemset(&buffers[2], 0, 100 * 81 * 28 * 28 * sizeof(float));

		cudaMemcpy(buffers[0], h_input_0, INPUT_SIZE * INPUT_SIZE * 3 * sizeof(float), cudaMemcpyHostToDevice);

		// -- do execute --------//
		//engine_context->executeV2(buffers);   //有implictDim的Error,需要显式指定batch
		engine_context->execute(1,buffers);

		memset(h_output_box, 0, 100 * 6 * sizeof(float));
		memset(h_output_mask,0, 100 * 81 * 28 * 28 * sizeof(float));

		cudaMemcpy(h_output_box, buffers[1], 100 * 6 * sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(h_output_mask, buffers[2], 100 * 81 * 28 * 28 * sizeof(float), cudaMemcpyDeviceToHost);


		std::vector<BBoxInfo>  bbinfos;
		std::vector<float> imginfo = { h,w };
		decodeOutput(imginfo, h_output_box, h_output_mask, &bbinfos);

		drawBox(image_origin, bbinfos);

		int index = fn[i].find_last_of("\\");
		//Get filename with extension
		std::string filename = fn[i].substr(index + 1, -1);

		cv::imwrite("res/"+ filename, image_origin);

		//bbinfos.clear();
		bbinfos = std::vector<BBoxInfo>();
	
		cudaFree(buffers[0]);
		cudaFree(buffers[1]);
		cudaFree(buffers[2]);
	
	}

	delete[] h_input_0;
	delete[] h_output_box;
	delete[] h_output_mask;

	engine_runtime->destroy();
	engine_infer->destroy();

	return 0;
}




