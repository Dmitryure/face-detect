#include <string>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include "utils.h"
#include "vector"

using namespace std;

void show(string window_name, cv::Mat img)
{
    cv::namedWindow(window_name, cv::WINDOW_AUTOSIZE);
    cv::moveWindow(window_name, 700, 150);
    cv::imshow(window_name, img);
    cv::waitKey(0);
};

string video_path(string path)
{
    cout << path << endl;
    return "./videos/" + path;
}

vector<cv::Mat> load_images(string name)
{
    vector<cv::Mat> images;

    for (size_t i = 1; i < 11; i++)
    {
        cv::Mat image = cv::imread("./images/" + name + "_" + to_string(i) + ".jpg", cv::IMREAD_GRAYSCALE);
        images.push_back(image);
        cout << "./images/" + name + "_" + to_string(i) + ".jpg" + " READ" << endl;
    };
    cout << "images" << images.size() << endl;
    return images;
}