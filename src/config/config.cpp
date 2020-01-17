#include "config.hpp"

Config::Config(std::string fileName) {
    std::ifstream i(fileName);
    json j;
    i >> j;
    // even easier with structured bindings (C++17)
    token =  j.at("telegram").at("token"); 
    yolo.push_back(j.at("yolo").at("model"));
    yolo.push_back(j.at("yolo").at("config"));
    yolo.push_back(j.at("yolo").at("classes"));   
}

Config::~Config() {

}