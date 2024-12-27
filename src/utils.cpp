#include <string>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include "utils.h"
#include "vector"
#include "map"
#include <regex>
#include <filesystem>
#include <fstream>

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
    std::cout << path << endl;
    return "./videos/" + path;
}

std::string extractBaseName(const std::string &filename)
{
    // Regular expression to match "foo_bar" part
    regex pattern(R"((.*)_[0-9]+\.[^\.]+$)");
    smatch match;

    if (regex_match(filename, match, pattern))
    {
        return match[1]; // Return the first capturing group
    }

    // If no match, return the original string
    return filename;
}

map<string, vector<cv::Mat>> load_images()
{
    map<string, vector<cv::Mat>> images;
    std::string imageFolder = "images";
    for (const auto &entry : std::filesystem::directory_iterator(imageFolder))
    {

        std::string fileName = entry.path().filename().string();
        std::string filePath = entry.path().string();

        // Read the image
        cv::Mat image = cv::imread(filePath, cv::IMREAD_GRAYSCALE);
        if (image.empty())
        {
            std::cerr << "Warning: Unable to read image file " << fileName << std::endl;
            continue;
        }
        cv::normalize(image, image, 0, 255, cv::NORM_MINMAX);
        cv::Mat mirrored;
        cv::Mat blurred;
        cv::flip(image, mirrored, 1);
        GaussianBlur(image, blurred, cv::Size(5, 5), 1.5);
        std::string actorName;
        actorName = extractBaseName(fileName);
        images[actorName].push_back(image);
        images[actorName].push_back(mirrored);
        images[actorName].push_back(blurred);
    }

    return images;
}

std::vector<cv::VideoCapture> load_videos()
{
    std::vector<cv::VideoCapture> videos;
    std::string videosFolder = "videos";
    for (const auto &entry : std::filesystem::directory_iterator(videosFolder))
    {

        std::string filePath = entry.path().string();

        cv::VideoCapture video = cv::VideoCapture(filePath);
        videos.push_back(video);
    }

    return videos;
}
