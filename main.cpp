#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/line_descriptor.hpp>
#include <opencv2/ximgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/features2d.hpp>
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

double euclideanDistance(const cv::Mat &vec1, const cv::Mat &vec2)
{
    return cv::norm(vec1, vec2, cv::NORM_L2); // L2 norm (Euclidean distance)
}

cv::Mat computeHOG(const cv::Mat &img, cv::HOGDescriptor &hog)
{
    std::vector<float> descriptors;
    hog.compute(img, descriptors);
    return cv::Mat(descriptors).clone();
}

int main()
{
    show_files();
    cv::Mat photo = cv::imread("./images/woman.png", cv::IMREAD_GRAYSCALE);
    VideoCapture vid = VideoCapture(video_path("kurt_russel_china.avi"));
    Mat frame;
    std::vector<cv::Ptr<cv::Tracker>> trackers;
    std::vector<Rect> faces;

    for (;;)
    {
        vid >> frame;

        if (frame.empty())
            break;
        Mat frame_gray;
        cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
        int frameIndex = static_cast<int>(vid.get(cv::CAP_PROP_POS_FRAMES));
        if (frameIndex % 10 == 0)
        {
            faces = detectAndDisplay(frame_gray);
            if (faces.size())
            {
                trackers.clear();
                for (size_t face = 0; face < faces.size(); face++)
                {
                    auto tracker = cv::TrackerKCF::create();
                    cv::Rect myROI(faces[face].x, faces[face].y, faces[face].width, faces[face].height);
                    tracker->init(frame, myROI);
                    // Resize the ROI to match the HOG descriptor window size
                    trackers.push_back(tracker);
                }
            };
        };
        if (!faces.size())
            continue;
        for (size_t i = 0; i < faces.size(); i++)
        {
            cv::Rect2d myROI(faces[i].x, faces[i].y, faces[i].width, faces[i].height);
            bool success = trackers[i]->update(frame, myROI);

            if (success)
            {
                cv::rectangle(frame, myROI, cv::Scalar(0, 255, 0), 2, 1);
            }
            else
            {
                std::cerr << "Tracking failure for ROI " << i << std::endl;
            }
        }

        // cv::Rect myROI(10, 10,100, 100);
        show("Test", frame);
    }
    return 0;
}