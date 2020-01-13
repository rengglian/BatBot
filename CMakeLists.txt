cmake_minimum_required(VERSION 3.16.2)
set(TARGET_NAME batbot)
set(TARGET_VERSION 0.1)
project(${TARGET_NAME} VERSION ${TARGET_VERSION})

set(CMAKE_CXX_STANDARD 17)
set(Boost_USE_MULTITHREADED ON)

find_package(Threads REQUIRED)
find_package(OpenSSL REQUIRED)
find_package(Boost COMPONENTS system REQUIRED)
find_package(CURL)
find_package(OpenCV REQUIRED)

add_executable(${TARGET_NAME}   src/main.cpp 
                                src/bot/bot.cpp
                                src/detection/detection.cpp)

target_include_directories(${TARGET_NAME} PRIVATE ${OPENSSL_INCLUDE_DIR} ${Boost_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS})

if (CURL_FOUND)
    target_include_directories(${TARGET_NAME} PRIVATE ${CURL_INCLUDE_DIRS})
endif()

target_include_directories(${TARGET_NAME} PUBLIC    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>/src/bot
                                                    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>/src/detection

)


target_link_libraries(${TARGET_NAME} /usr/local/lib/libTgBot.a ${CMAKE_THREAD_LIBS_INIT} ${OPENSSL_LIBRARIES} ${Boost_LIBRARIES} ${CURL_LIBRARIES} ${OpenCV_LIBS})
