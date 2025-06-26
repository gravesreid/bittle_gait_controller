#include <fcntl.h>
#include <termios.h>
#include <unistd.h>
#include <iostream>
#include <cstring>

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
    std::cout << "Sent: " << command << std::endl;
}

int main() {
    const char* port = "/dev/rfcomm0"; // Update to your Bittle's port
    int fd = openSerialPort(port);
    if (fd < 0) return 1;

    // Set terminal to non-blocking input
    termios oldt, newt;
    tcgetattr(STDIN_FILENO, &oldt);
    newt = oldt;
    newt.c_lflag &= ~(ICANON | ECHO);
    tcsetattr(STDIN_FILENO, TCSANOW, &newt);

    std::cout << "Controls:\n"
              << "  W - Forward\n"
              << "  A - Turn Left\n"
              << "  S - Backward\n"
              << "  D - Turn Right\n"
              << "  SPACE - Stop\n"
              << "  Q - Quit\n\n"
              << "Press keys to control Bittle...\n";

    char key;
    while (true) {
        fd_set set;
        FD_ZERO(&set);
        FD_SET(STDIN_FILENO, &set);
        timeval timeout = {0, 10000};  // 10ms

        if (select(1, &set, NULL, NULL, &timeout) > 0) {
            read(STDIN_FILENO, &key, 1);
            key = tolower(key);

            switch (key) {
                case 'w': sendCommand(fd, "kwkF"); break;  // Walk forward
                case 'a': sendCommand(fd, "kvtL"); break;  // Turn left
                case 's': sendCommand(fd, "kbk");  break;  // Walk backward
                case 'd': sendCommand(fd, "kvtR"); break;  // Turn right
                case ' ': sendCommand(fd, "kbalance"); break;  // Stop
                case 'q': 
                    tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
                    close(fd);
                    return 0;
                default: break;
            }
        }
    }
}