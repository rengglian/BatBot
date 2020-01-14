#include "config.hpp"

Config::Config() {
    
}

Config::~Config() {

}

void Config::Read(std::string fileName)
{
    std::ifstream i(fileName);
    json j;
    i >> j;
    // even easier with structured bindings (C++17)
    token =  j.at("telegram").at("token");
    
}