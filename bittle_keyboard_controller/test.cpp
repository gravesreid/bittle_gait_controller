#include <fcntl.h>
#include <termios.h>
#include <unistd.h>
#include <iostream>
#include <cstring>
#include "xbox_controller.h"
#include <linux/input.h>  // Missing for input_event, EV_KEY, etc.
#include <sys/select.h>   // Missing for select()
#include <errno.h>        // Missing for errno

int openSerialPort(const char* port) {
    int fd = open(port, O_RDWR | O_NOCTTY | O_NDELAY);
    if (fd == -1) {
        std::cerr << "Error opening serial port " << port << std::endl;
        return -1;
    }

    struct termios tty;
    memset(&tty, 0, sizeof(tty));
    if (tcgetattr(fd, &tty) != 0) {
        std::cerr << "Error getting serial attributes" << std::endl;
        close(fd);
        return -1;
    }

    cfsetospeed(&tty, B115200);
    cfsetispeed(&tty, B115200);

    tty.c_cflag &= ~PARENB;    // No parity
    tty.c_cflag &= ~CSTOPB;    // 1 stop bit
    tty.c_cflag &= ~CSIZE;
    tty.c_cflag |= CS8;        // 8-bit data
    tty.c_cflag &= ~CRTSCTS;   // No hardware flow control
    tty.c_cflag |= CREAD | CLOCAL;  // Enable reading

    tty.c_lflag &= ~ICANON;    // Non-canonical mode
    tty.c_lflag &= ~ECHO;      // No echo
    tty.c_lflag &= ~ECHOE;
    tty.c_lflag &= ~ECHONL;
    tty.c_lflag &= ~ISIG;      // No signal chars

    tty.c_iflag &= ~(IXON | IXOFF | IXANY); // No software flow control
    tty.c_iflag &= ~(IGNBRK | BRKINT | PARMRK | ISTRIP | INLCR | IGNCR | ICRNL);

    tty.c_oflag &= ~OPOST;     // Raw output
    tty.c_oflag &= ~ONLCR;

    tty.c_cc[VTIME] = 10;     // 1 second timeout (deciseconds)
    tty.c_cc[VMIN] = 0;

    if (tcsetattr(fd, TCSANOW, &tty) != 0) {
        std::cerr << "Error setting serial attributes" << std::endl;
        close(fd);
        return -1;
    }

    return fd;
}

void sendCommand(int fd, const char* command) {
    std::string fullCommand = std::string(command) + "\r";
    write(fd, fullCommand.c_str(), fullCommand.size());
    // Flush the output buffer immediately
    tcdrain(fd);  // Wait for all data to be transmitted
    std::cout << "Sent: " << command << std::endl;
}

int main() {
    const char* port = "/dev/rfcomm0"; // Update to your Bittle's port
    int serial_fd = openSerialPort(port);
    if (serial_fd < 0) return 1;
    int xbox_fd = 0;
    // Set terminal to non-blocking input
    termios oldt, newt;
    tcgetattr(STDIN_FILENO, &oldt);
    newt = oldt;
    newt.c_lflag &= ~(ICANON | ECHO);
    tcsetattr(STDIN_FILENO, TCSANOW, &newt);

    // Now you can use Xbox controller functions
    std::string xbox_device = find_xbox_device();
    if (!xbox_device.empty()) {
        std::cout << "Found Xbox controller at: " << xbox_device << std::endl;
        xbox_fd = open(xbox_device.c_str(), O_RDONLY |O_NONBLOCK);
        if (xbox_fd >= 0) {
            std::cout << "Controller name: " << get_device_name(xbox_fd) << std::endl;
            std::cout << "Press Xbox button to exit" << std::endl;
        }
    } else { 
        std::cerr << "Xbox controller not found!" << std::endl;
        close(serial_fd);
        return 1;
    }

    
    
    
    int xcommand = -1; // 0 = left, 1 = right, -1 = neutral;
    int ycommand = -1 ;// 0 = forward, 1 = backward, -1 = neutral;
    bool quit = false;
    const int DEADZONE = 12000;  // Adjust based on controller sensitivity
    
    enum Direction { 
        BALANCE = 0, 
        FORWARD = 1, 
        BACKWARDS = 2,
        FRONT_LEFT = 3,
        FRONT_RIGHT = 4,
        BACK_LEFT = 5,
        BACK_RIGHT = 6,
        LEFT = 7,
        RIGHT = 8,
        JUMP = 9
    };
    Direction dir = BALANCE;
    Direction lastDir = BALANCE;
    input_event ev;     // Missing declaration

    // Add this before your main loop
    int current_x = 32767;  // Store current X position
    int current_y = 32767;  // Store current Y position
    bool axis_changed = false;  // Flag to track if any axis changed
    while (!quit) {
        fd_set fds;
        FD_ZERO(&fds);
        FD_SET(xbox_fd, &fds);
        
        // Set timeout for select
        timeval timeout;
        timeout.tv_sec = 0;
        timeout.tv_usec = 100000;  // 100ms
        
        int ret = select(xbox_fd + 1, &fds, NULL, NULL, &timeout);
        if (ret < 0) {
            perror("select error");
            break;
        }
        
        if (ret > 0) {
            bool axis_changed = false;
            
            // Process all available events
            while (read(xbox_fd, &ev, sizeof(ev)) == sizeof(ev)) {
                // Check for quit button

                if (ev.type == EV_KEY && ev.code == BTN_B && ev.value == 1) {
                    quit = true;
                    break;
                }
                if (ev.type == EV_KEY && ev.code == BTN_A && ev.value == 1) {
                    std::cout << "Jumping...\n";
                    sendCommand(serial_fd, "kjmp");
                    // for (size_t i = 0; i < 5; i++)
                    // {
                    //     sendCommand(serial_fd, "kbalance");
                    // }
                }
                if (ev.type == EV_ABS && ev.code == 9) {
                    std::cout << "Braking...\n";
                    for (size_t i = 0; i < 2; i++)
                    {
                        sendCommand(serial_fd, "kbalance");
                    }
                }
                
                // Process joystick movements
                if (ev.type == EV_ABS) {
                    // Left stick X-axis (horizontal)
                    if (ev.code == ABS_X) {
                        if (ev.value < (32767 - DEADZONE)) {
                            xcommand = 0;  // Left
                        } else if (ev.value > (32767 + DEADZONE)) {
                            xcommand = 1;  // Right
                        } else {
                            xcommand = -1; // Neutral
                        }
                        axis_changed = true;
                    }
                    
                    // Left stick Y-axis (vertical)
                    if (ev.code == ABS_Y) {
                        if (ev.value < (32767 - DEADZONE)) {
                            ycommand = 0;  // Forward
                        } else if (ev.value > (32767 + DEADZONE)) {
                            ycommand = 1;  // Backward
                        } else {
                            ycommand = -1; // Neutral
                        }
                        axis_changed = true;
                    }
                }
            }

                // Only calculate direction and send commands if axis values changed
                if (axis_changed) {
                    // Calculate direction based on combined axis values
                    if (xcommand == 0 && ycommand == 0) {
                        dir = FRONT_LEFT;         // Forward + Left
                    } else if (xcommand == 1 && ycommand == 0) {
                        dir = FRONT_RIGHT;        // Forward + Right
                    } else if (xcommand == 0 && ycommand == -1) {
                        dir = LEFT;               // Left only
                    } else if (xcommand == 1 && ycommand == -1) {
                        dir = RIGHT;              // Right only
                    } else if (xcommand == -1 && ycommand == 0) {
                        dir = FORWARD;            // Forward only
                    } else if (xcommand == -1 && ycommand == 1) {
                        dir = BACKWARDS;          // Backward only
                    } else if (xcommand == 1 && ycommand == 1) {
                        dir = BACK_RIGHT;         // Backward + Right
                    } else if (xcommand == 0 && ycommand == 1) {
                        dir = BACK_LEFT;          // Backward + Left
                    } else {
                        dir = BALANCE;            // Neutral/Center
                    }
            
                    // Send commands only if direction changed
                    if (dir != lastDir) {
                        switch (dir) {
                            case BALANCE:
                                sendCommand(serial_fd, "kbalance");
                                break;
                            case FORWARD:
                                sendCommand(serial_fd, "ktrF");
                                break;
                            case BACKWARDS:
                                sendCommand(serial_fd, "kbk");
                                break;
                            case FRONT_LEFT:
                                sendCommand(serial_fd, "ktrL");
                                break;
                            case FRONT_RIGHT:
                                sendCommand(serial_fd, "ktrR");
                                break;
                            case BACK_LEFT:
                                sendCommand(serial_fd, "kbkL");
                                break;
                            case BACK_RIGHT:
                                sendCommand(serial_fd, "kbkR");
                                break;
                            case LEFT:
                                sendCommand(serial_fd, "kvtL");
                                break;
                            case RIGHT:
                                sendCommand(serial_fd, "kvtR");
                                break;
                            case JUMP:
                                sendCommand(serial_fd, "kjmp");
                                break;
                        }
                        lastDir = dir;
                    }
            }
        }
    }
    sendCommand(serial_fd, "kbalance");
    close(xbox_fd);
    close(serial_fd);
    std::cout << "Exiting cleanly\n";
    return 0;
    
}