#include "detection.hpp"

using namespace cv;
using namespace dnn;

ObjectDetection::ObjectDetection(std::shared_ptr<Config> config) : config_(config) {
    std::vector<std::string> configFiles = config_->GetYolo();
    std::string modelFile = configFiles[0];
    std::string configFile = configFiles[1];
    std::string classesFile = configFiles[2];

    loadNet(modelFile, configFile);
    loadClasses(classesFile);
}

void ObjectDetection::loadClasses(std::string classesFile){
    std::ifstream ifs(classesFile.c_str());
    if (!ifs.is_open())
        CV_Error(Error::StsError, "File " + classesFile + " not found");
    std::string line;
    while (std::getline(ifs, line))
    {
        classes.push_back(line);
    }
}

void ObjectDetection::loadNet(std::string modelFile, std::string configFile){
    // Load a model.
    net_ = readNet(modelFile, configFile, ""); 
    net_.setPreferableBackend(0);
    net_.setPreferableTarget(0);  
}

ObjectDetection::~ObjectDetection() {

}

std::unordered_map<std::string,int> ObjectDetection::detection(std::vector<uint8_t>& vec)
{
    float scale = 1.0/255.0;
    Scalar mean;
    bool swapRB = true;
    int inpWidth = 416;
    int inpHeight = 416;

    std::vector<String> outNames = net_.getUnconnectedOutLayersNames();

    // Process frames.
    Mat blob;
    Mat data_mat(vec,true);
    Mat frame(cv::imdecode(data_mat,1)); //put 0 if you want greyscale
    
    std::unordered_map<std::string,int> results;

    if (!frame.empty()) {
        
        preprocess(frame, net_, Size(inpWidth, inpHeight), scale, mean, swapRB);

        std::vector<Mat> outs;
        net_.forward(outs, outNames);

        auto objects = postprocess(frame, outs, net_);

        results = updateFrame(frame, objects);

        imencode(".jpg", frame, vec);
    }
    return results;
}

void ObjectDetection::preprocess(const Mat& frame, Net& net, Size inpSize, float scale,
                       const Scalar& mean, bool swapRB)
{
    static Mat blob;
    // Create a 4D blob from a frame.
    if (inpSize.width <= 0) inpSize.width = frame.cols;
    if (inpSize.height <= 0) inpSize.height = frame.rows;
    blobFromImage(frame, blob, 1.0, inpSize, Scalar(), swapRB, false, CV_8U);

    // Run a model.
    net.setInput(blob, "", scale, mean);    
}

std::vector<ObjectDetection::object> ObjectDetection::postprocess(const Mat& frame, const std::vector<Mat>& outs, Net& net)
{
    static std::vector<int> outLayers = net.getUnconnectedOutLayers();
    static std::string outLayerType = net.getLayer(outLayers[0])->type;

    std::vector<String> outNames = net.getUnconnectedOutLayersNames();

	std::vector<ObjectDetection::object>Objects;

    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<Rect> boxes;
    if (outLayerType == "DetectionOutput")
    {
        // Network produces output blob with a shape 1x1xNx7 where N is a number of
        // detections and an every detection is a vector of values
        // [batchId, classId, confidence, left, top, right, bottom]
        CV_Assert(outs.size() > 0);
        for (size_t k = 0; k < outs.size(); k++)
        {
            float* data = (float*)outs[k].data;
            for (size_t i = 0; i < outs[k].total(); i += 7)
            {
                float confidence = data[i + 2];
                if (confidence > confThreshold)
                {
                    int left   = (int)data[i + 3];
                    int top    = (int)data[i + 4];
                    int right  = (int)data[i + 5];
                    int bottom = (int)data[i + 6];
                    int width  = right - left + 1;
                    int height = bottom - top + 1;
                    if (width <= 2 || height <= 2)
                    {
                        left   = (int)(data[i + 3] * frame.cols);
                        top    = (int)(data[i + 4] * frame.rows);
                        right  = (int)(data[i + 5] * frame.cols);
                        bottom = (int)(data[i + 6] * frame.rows);
                        width  = right - left + 1;
                        height = bottom - top + 1;
                    }

                    ObjectDetection::object o;
                    o.classId = (int)(data[i + 1]) - 1;
                    o.box = Rect(left, top, width, height);
                    o.confidence = confidence;
                    Objects.push_back(o);

                    classIds.push_back((int)(data[i + 1]) - 1);  // Skip 0th background class id.
                    boxes.push_back(Rect(left, top, width, height));
                    confidences.push_back(confidence);
                }
            }
        }
    }
    else if (outLayerType == "Region")
    {

        for (size_t i = 0; i < outs.size(); ++i)
        {
            // Network produces output blob with a shape NxC where N is a number of
            // detected objects and C is a number of classes + 4 where the first 4
            // numbers are [center_x, center_y, width, height]
            float* data = (float*)outs[i].data;
            for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
            {
                
                Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
                Point classIdPoint;
                double confidence;
                minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
                if (confidence > confThreshold)
                {
                    int centerX = (int)(data[0] * frame.cols);
                    int centerY = (int)(data[1] * frame.rows);
                    int width = (int)(data[2] * frame.cols);
                    int height = (int)(data[3] * frame.rows);
                    int left = centerX - width / 2;
                    int top = centerY - height / 2;

                    ObjectDetection::object o;
                    o.classId = classIdPoint.x;
                    o.box = Rect(left, top, width, height);
                    o.confidence = confidence;
                    Objects.push_back(o);

                    classIds.push_back(classIdPoint.x);  // Skip 0th background class id.
                    boxes.push_back(Rect(left, top, width, height));
                    confidences.push_back(confidence);
                    //std::cout << confidence << "\t" << centerX << "\t"<< centerY << "\t" << width << "\t" << height << "\t" << classes[classIdPoint.x] << std::endl;
                }
            }
        }
    } else {
        CV_Error(Error::StsNotImplemented, "Unknown output layer type: " + outLayerType);
    }

    return Objects;
}

std::unordered_map<std::string,int> ObjectDetection::updateFrame(Mat& frame, std::vector<ObjectDetection::object> objects) 
{
    static auto val = time(0);   
    srand((unsigned) val);
    const int min = 128;
    const int max = 255;

    std::unordered_map<std::string, int> results;
    std::unordered_map<std::string, Scalar> classColor;
    
    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<Rect> boxes;

    for (const auto& object : objects)
    {  
        classIds.push_back(object.classId);  // Skip 0th background class id.
        boxes.push_back(object.box);
        confidences.push_back(object.confidence);
    }
    
    std::vector<int> indices;
    NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
    for (size_t i = 0; i < indices.size(); ++i)
    {
        int idx = indices[i];
        results[classes[classIds[idx]]]++;
        auto it = classColor.find(classes[classIds[idx]]);
        if (it == classColor.end()) 
        {
            int R = min + (rand() % (max-min));
            int G = min + (rand() % (max-min));
            int B = min + (rand() % (max-min));
            
            classColor[classes[classIds[idx]]] = Scalar(R, G, B);
        }
        Rect box = boxes[idx];

        drawPred(classIds[idx],classColor[classes[classIds[idx]]], confidences[idx], box.x, box.y,
                box.x + box.width, box.y + box.height, frame);
    }  
    return results;
}

void ObjectDetection::drawPred(int classId, Scalar classColor, float conf, int left, int top, int right, int bottom, Mat& frame)
{
    std::string label = format("%.2f", conf);
    if (!classes.empty())
    {
        CV_Assert(classId < (int)classes.size());
        label = classes[classId] + ": " + label;
    }
    int baseLine;
    
    Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    top = max(top, labelSize.height);
    rectangle(frame, Point(left, top), Point(right, bottom), classColor, 2);
    rectangle(frame, Point(left, top - labelSize.height),
              Point(left + labelSize.width, top + baseLine), classColor, FILLED);
    putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.5, Scalar());
}