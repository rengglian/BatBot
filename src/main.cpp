#include <iostream>
#include <fstream>
#include "bot/bot.hpp"

int main(int argc, char* argv[]) {
	
	//Create Bot instance
	Bot bot;

	//Set up telegram Bot
	bot.SetUpCommands();

    //Set up telegram Bot
	bot.SetUpMessages();
	
	//Loop
	bot.Listen();

	return 0;
}