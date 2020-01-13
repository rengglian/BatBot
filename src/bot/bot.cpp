#include "bot.hpp"

Bot::Bot() {
    
	bot = new TgBot::Bot("1055954257:AAGX3OdfgWVxMvdAeRwazLXlRgJjnipi4xY");
}

Bot::~Bot() {
	delete bot;
}

void Bot::Listen() {
	while (true) {
		try {
			std::clog << "Bot username: " << bot->getApi().getMe()->username.c_str() << std::endl;

			TgBot::TgLongPoll longPoll(*bot);

			std::clog << "Long poll started" << std::endl;
			while (true) {
				try {
					longPoll.start();
				} catch (std::runtime_error& e) {
					std::cerr << "Error (listen): " << e.what() << std::endl;
				}
			}
		} catch (std::runtime_error& e) {
			std::cerr << "Error (longPoll): " << e.what() << std::endl;
		}
	}
}

void Bot::SetUpCommands() {
	bot->getEvents().onCommand("start", [this](TgBot::Message::Ptr message) {
		bot->getApi().sendMessage(message->chat->id, "Hi!");
	});
}

void Bot::SetUpMessages() {
    bot->getEvents().onAnyMessage([this](TgBot::Message::Ptr  message) {
        time_t ltime;
        struct tm result;
        char stime[32];

        ltime = time(NULL);
        localtime_r(&ltime, &result);
        asctime_r(&result, stime);
        std::clog << stime << "\t" << message->from->username.c_str() << "\t" << message->text.c_str() << std::endl;
        if (StringTools::startsWith(message->text, "/start")) {
            return;
        }
        if (!message->photo.empty()){
            auto photo = message->photo.back();
            TgBot::File::Ptr pfile = bot->getApi().getFile(photo->fileId);
            std::string imgStr = bot->getApi().downloadFile(pfile->filePath);
            std::vector<uint8_t> vec(imgStr.begin(), imgStr.end());
        
            ObjectDetection yolo;
            auto result = yolo.detection(vec);

            std::string caption = "Image contains:\n";

            for (auto &e: result) {
		        caption += e.first + " " + std::to_string(e.second) + "\n";
	        }

            auto image(std::make_shared<TgBot::InputFile>());
            std::string tmp(vec.begin(), vec.end()); 
            image->data = tmp;
            image->mimeType = "image/jpeg";
            image->fileName = "Hello.jpg";
            bot->getApi().sendPhoto(message->chat->id, image, caption);
        } else { 
            bot->getApi().sendMessage(message->chat->id, "Your message is: " + message->text);
        }
    });
}

std::string Bot::GetUsername(TgBot::User::Ptr user) {
	return user->firstName + " " + user->lastName + " (@"+user->username+")";
}