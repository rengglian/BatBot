#include <iostream>
#include <fstream>
#include "bot/bot.hpp"
#include "config/config.hpp"

int main(int argc, char* argv[]) {
	
	//Config config;
	std::shared_ptr<Config> config_ = std::make_shared<Config>(argv[1]);
	std::shared_ptr<ObjectDetection> yolo_ = std::make_shared<ObjectDetection>(config_);
	std::shared_ptr<Bot> bot_ = std::make_shared<Bot>(config_, yolo_);

            

	//Set up telegram Bot
	bot_->SetUpCommands();

    //Set up telegram Bot
	bot_->SetUpMessages();
	
	//Loop
	bot_->Listen();

	return 0;
}