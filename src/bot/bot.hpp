#pragma once

#include <time.h>
#include <tgbot/tgbot.h>

#include "../config/config.hpp"
#include "../detection/detection.hpp"

class Bot {

	public:

		Bot(std::string token);
		~Bot();
		
		void Listen();

		//Bot methods
		void SetUpCommands();
        void SetUpMessages();

	private:

		TgBot::Bot *bot;

		//Telegram methods
		std::string GetUsername(TgBot::User::Ptr user);
};