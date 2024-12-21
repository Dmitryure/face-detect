#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/ximgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/tracking/tracker.hpp>
#include "utils.h"
#include "detectors.h"
#include <dirent.h> // POSIX library for directory traversal
using namespace std;

using namespace cv;

void show(string window_name, cv::Mat img)
{
    cv::namedWindow(window_name, cv::WINDOW_AUTOSIZE);
    cv::moveWindow(window_name, 700, 150);
    cv::imshow(window_name, img);
    cv::waitKey(0);
};

int show_files()
{

    // Get the current directory (can be set explicitly if needed)
    std::string currentDir = ".";

    // Open the directory
    DIR *dir = opendir(currentDir.c_str());
    if (dir == nullptr)
    {
        std::cerr << "Error: Unable to open directory " << currentDir << "\n";
        return 1;
    }

    std::cout << "Files in the current directory:\n";

    // Read the directory entries
    struct dirent *entry;
    while ((entry = readdir(dir)) != nullptr)
    {
        std::string name = entry->d_name;
        if (name == "." || name == "..")
        {
            continue; // Skip current and parent directory entries
        }
        std::cout << " - " << name << "\n";
    }

    // Close the directory
    closedir(dir);
    return 0;
}

int main()
{
    show_files();
    VideoCapture vid = VideoCapture(video_path("kurt_russel_china.avi"));
    Mat frame;
    for (;;)
    {
        vid >> frame;

        if (frame.empty())
            break;
        std::vector<Rect> faces = detectAndDisplay(frame);
        if (!faces.size())
            continue;
        cv::Rect myROI(faces[0].x, faces[0].y, faces[0].width, faces[0].height);
        // cv::Rect myROI(10, 10,100, 100);
        show("Test", frame(myROI));
    }
    return 0;
}