#pragma once
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

class Config {

	public:

		Config();
		~Config();
		
		void Read(std::string fileName);
		std::string GetToken() {return token;}

	private:

	std::string token;
};