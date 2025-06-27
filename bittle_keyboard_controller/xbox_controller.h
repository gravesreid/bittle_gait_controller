#ifndef XBOX_CONTROLLER_H
#define XBOX_CONTROLLER_H

#include <string>

// Function declarations
std::string get_device_name(int fd);
std::string find_xbox_device();
const char* event_type(int type);
const char* axis_name(int code);
const char* button_name(int code);

#endif // XBOX_CONTROLLER_H