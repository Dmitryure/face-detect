#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/ximgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/tracking/tracker.hpp>
#include "utils.h"
#include "detectors.h"
using namespace std;
using namespace cv;

void show(string window_name, cv::Mat img)
{
    cv::namedWindow(window_name, cv::WINDOW_AUTOSIZE);
    cv::moveWindow(window_name, 700, 150);
    cv::imshow(window_name, img);
    cv::waitKey(0);
};

int main()
{
    VideoCapture vid = VideoCapture(video_path("kurt_russel_china.avi"));
    Mat frame;
    for (;;)
    {
        vid >> frame;

        if (frame.empty())
            break;

        detectAndDisplay(frame);
        show("Test", frame);
    }
    return 0;
}