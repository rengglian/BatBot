#pragma once
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

class Config {

	public:

		Config(std::string fileName);
		~Config();
		
		void Read();

		std::string GetToken() {return token;};
		std::vector<std::string> GetYolo() {return yolo;};

	private:

	std::string token;
	std::vector<std::string> yolo;
};