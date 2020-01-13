#pragma once
#include <fstream>
#include <iostream>
#include <sstream>

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>

class ObjectDetection {

	public:

		ObjectDetection();
		~ObjectDetection();
		
		void detection(std::vector<uint8_t>& vec);


		struct item {
			int classId;
			float confidence;
			cv::Rect box;
		};

		 std::vector<item>Items;

	private:

		void postprocess(cv::Mat& frame, const std::vector<cv::Mat>& outs, cv::dnn::Net& net);
        void preprocess(const cv::Mat& frame, cv::dnn::Net& net, cv::Size inpSize, float scale, const cv::Scalar& mean, bool swapRB);
		void drawPred(int classId, float conf, int left, int top, int right, int bottom, cv::Mat& frame);

        std::vector<std::string> classes;                       
};