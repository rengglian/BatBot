#pragma once
#include <fstream>
#include <iostream>
#include <sstream>
#include <unordered_map>
#include <ctime>

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>

#include "../config/config.hpp"

class ObjectDetection {

	public:

		ObjectDetection(std::shared_ptr<Config> config);
		~ObjectDetection();
		
		std::unordered_map<std::string,int> detection(std::vector<uint8_t>& vec);

		struct object {
			int classId;
			float confidence;
			cv::Rect box;
		};

	private:

    	const float confThreshold = 0.5;
    	const float nmsThreshold = 0.4;

		void loadClasses(std::string classFile);
		void loadNet(std::string modelFile, std::string configFile);

		std::vector<object> postprocess(const cv::Mat& frame, const std::vector<cv::Mat>& outs, cv::dnn::Net& net);
        void preprocess(const cv::Mat& frame, cv::dnn::Net& net, cv::Size inpSize, float scale, const cv::Scalar& mean, bool swapRB);
		std::unordered_map<std::string,int> updateFrame(cv::Mat& frame, const std::vector<object> objects);
		void drawPred(int classId, cv::Scalar classColor, float conf, int left, int top, int right, int bottom, cv::Mat& frame);

        std::vector<std::string> classes;
		cv::dnn::Net net_;

		const std::shared_ptr<Config> config_;                    
};