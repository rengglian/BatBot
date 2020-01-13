#pragma once

#include <time.h>
#include <tgbot/tgbot.h>

#include "../detection/detection.hpp"

class Bot {

	public:

		Bot();
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