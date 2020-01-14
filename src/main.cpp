#include <iostream>
#include <fstream>
#include "bot/bot.hpp"
#include "config/config.hpp"

int main(int argc, char* argv[]) {
	

	Config config;
	config.Read("./config/config.json");

	//Create Bot instance
	Bot bot(config.GetToken());

	//Set up telegram Bot
	bot.SetUpCommands();

    //Set up telegram Bot
	bot.SetUpMessages();
	
	//Loop
	bot.Listen();

	return 0;
}