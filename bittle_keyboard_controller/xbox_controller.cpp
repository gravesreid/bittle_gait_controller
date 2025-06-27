#include <fcntl.h>
#include <unistd.h>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>
#include <cstring>
#include <sys/ioctl.h>
#include <linux/input.h>
#include <dirent.h>

// Get device name from file descriptor
std::string get_device_name(int fd) {
    char name[256] = "Unknown";
    ioctl(fd, EVIOCGNAME(sizeof(name)), name);
    return std::string(name);
}

// Find Xbox controller device path
std::string find_xbox_device() {
    const std::string input_dir = "/dev/input/";
    DIR* dir = opendir(input_dir.c_str());
    if (!dir) {
        perror("Failed to open input directory");
        return "";
    }

    dirent* entry;
    while ((entry = readdir(dir))) {
        std::string name = entry->d_name;
        // Check if it's an event device
        if (name.substr(0, 5) == "event") {
            std::string path = input_dir + name;
            int fd = open(path.c_str(), O_RDONLY | O_NONBLOCK);
            if (fd < 0) continue;

            std::string dev_name = get_device_name(fd);
            close(fd);

            // Check for known Xbox controller identifiers
            if (dev_name.find("Xbox") != std::string::npos ||
                dev_name.find("X-Box") != std::string::npos) {
                closedir(dir);
                return path;
            }
        }
    }

    closedir(dir);
    return "";
}

// Convert event type to string
const char* event_type(int type) {
    switch (type) {
        case EV_KEY: return "Button";
        case EV_ABS: return "Axis  ";
        case EV_SYN: return "Sync  ";
        default:     return "Other ";
    }
}

// Convert axis code to string
const char* axis_name(int code) {
    switch (code) {
        case ABS_X:  return "Left Stick X";
        case ABS_Y:  return "Left Stick Y";
        case ABS_RX: return "Right Stick X";
        case ABS_RY: return "Right Stick Y";
        case ABS_Z:  return "Left Trigger";
        case ABS_RZ: return "Right Trigger";
        case ABS_HAT0X: return "D-Pad X";
        case ABS_HAT0Y: return "D-Pad Y";
        default:     return "Unknown Axis";
    }
}

// Convert button code to string
const char* button_name(int code) {
    switch (code) {
        case BTN_A: return "A";
        case BTN_B: return "B";
        case BTN_X: return "X";
        case BTN_Y: return "Y";
        case BTN_TL: return "Left Bumper";
        case BTN_TR: return "Right Bumper";
        case BTN_SELECT: return "View";
        case BTN_START: return "Menu";
        case BTN_MODE: return "Xbox Button";
        case BTN_THUMBL: return "Left Stick Press";
        case BTN_THUMBR: return "Right Stick Press";
        default: return "Unknown Button";
    }
}

// int main() {
//     // Find Xbox controller
//     std::string dev_path = find_xbox_device();
//     if (dev_path.empty()) {
//         std::cerr << "Xbox controller not found!" << std::endl;
//         return 1;
//     }

//     std::cout << "Using device: " << dev_path << std::endl;
    
//     // Open device
//     int fd = open(dev_path.c_str(), O_RDONLY);
//     if (fd < 0) {
//         perror("Failed to open device");
//         return 1;
//     }

//     std::cout << "Xbox Controller detected: " << get_device_name(fd) << std::endl;
//     std::cout << "Press Ctrl+C to exit..." << std::endl << std::endl;

//     // Event reading loop
//     input_event ev;
//     while (true) {
//         ssize_t n = read(fd, &ev, sizeof(ev));
//         if (n != sizeof(ev)) {
//             if (errno == EAGAIN) continue;
//             perror("Error reading event");
//             break;
//         }

//         // Print event information
//         switch (ev.type) {
//             case EV_KEY:
//                 printf("[%s] %s: %s\n", 
//                        event_type(ev.type), 
//                        button_name(ev.code),
//                        ev.value ? "PRESSED " : "RELEASED");
//                 break;
                
//             case EV_ABS:
//                 // Special handling for D-Pad
//                 if (ev.code == ABS_HAT0X || ev.code == ABS_HAT0Y) {
//                     if (ev.value != 0) {  // Filter out neutral position
//                         printf("[%s] %s: %d\n", 
//                                event_type(ev.type), 
//                                axis_name(ev.code),
//                                ev.value);
//                     }
//                 } else {
//                     printf("[%s] %s: %d\n", 
//                            event_type(ev.type), 
//                            axis_name(ev.code),
//                            ev.value);
//                 }
//                 break;
//         }
//         fflush(stdout);
//     }

//     close(fd);
//     return 0;
// }