#pragma once

#include <time.h>
#include <tgbot/tgbot.h>
#include "../config/config.hpp"
#include "../detection/detection.hpp"

class Bot {

	public:

		Bot(std::shared_ptr<Config> config, std::shared_ptr<ObjectDetection> yolo);
		~Bot();
		
		void Listen();

		//Bot methods
		void SetUpCommands();
        void SetUpMessages();

	private:

		TgBot::Bot *bot;

		//Telegram methods
		std::string GetUsername(TgBot::User::Ptr user);

		const std::shared_ptr<Config> config_;
		const std::shared_ptr<ObjectDetection> yolo_;
};