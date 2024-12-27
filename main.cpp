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
#include <opencv2/face.hpp>

using namespace std;

using namespace cv;
using namespace cv::ml;

int show_files()
{

    // Get the current directory (can be set explicitly if needed)
    std::string currentDir = "./videos";

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
        std::cout << name.replace(name.begin() + name.size() - 4, name.end(), "") << "\n";
    }

    // Close the directory
    closedir(dir);
    return 0;
}

void ensureROIInBounds(cv::Rect2d &roi, const cv::Size &imageSize)
{
    // Ensure the ROI stays within the image boundaries
    roi.x = std::max(0, static_cast<int>(roi.x));
    roi.y = std::max(0, static_cast<int>(roi.y));
    roi.width = std::min(imageSize.width - roi.x, roi.width);
    roi.height = std::min(imageSize.height - roi.y, roi.height);
}

cv::Mat computeHOG(const cv::Mat &img, cv::HOGDescriptor &hog)
{
    std::vector<float> descriptors;
    hog.compute(img, descriptors);
    return cv::Mat(descriptors).clone();
}

int main()
{
    string tr;
    tr = "KCF";
    string algo;
    VideoCapture vid;
    cout << "Type \"knn\" for using KNN for recognition, \"facer\" for LBPHFaceRecognizer" << endl;
    cin >> algo;
    if (algo == "knn")
    {
        algo = "knn";
    }
    else
    {
        algo = "facer";
    }
    cout << algo << " selected" << endl;

    show_files();
    std::vector<cv::VideoCapture> videos;
    std::string video_name;
    cout << "g(1-4) = Gentlemen (2024), possible actors: kaya_scodelario, theo_james" << endl;
    cout << "pc(1-7) = Perfect Couple, The (2024), possible actors: nicole_kidman, liev_schreiber" << endl;
    cout << "lc(1-6) = Big Trouble in Little China (1986), possible actors: dennis_dun, kurt_russel" << endl;
    cout << "Either type \"all\" " << "or select video from above" << endl;
    cin >> video_name;
    if (video_name == "all")
    {
        videos = load_videos();
    }
    else
    {
        videos = {VideoCapture(video_path(video_name + ".avi"))};
    }
    cout << video_name << " selected" << endl;
    std::vector<cv::Ptr<cv::Tracker>> trackers;
    std::vector<Rect> faces;

    HOGDescriptor hog(Size(128, 256), Size(32, 32), Size(16, 16), Size(8, 8), 9);
    auto images = load_images();
    std::vector<int> labels; // Corresponding labels
    map<int, string> labelsMap;
    Mat trainingData, trainingLabels;
    std::vector<cv::Mat> recognizerTrainingData;
    int actorIndex = 1;
    for (auto &[key, vec] : images)
    {
        std::cout << "Processing category: " << key << std::endl;
        for (size_t index = 0; index < vec.size(); ++index)
        {
            cv::Mat &mat = vec[index];
            std::vector<float> descriptors;
            cv::resize(mat, mat, cv::Size(128, 256));
            if (algo == "knn")
            {

                hog.compute(mat, descriptors);
                trainingData.push_back(Mat(descriptors).t());
            }
            else
            {
                recognizerTrainingData.push_back(mat);
            }
            labels.push_back(actorIndex);
            labelsMap[actorIndex] = key;
            cout << actorIndex << key << endl;
        }
        actorIndex++;
    }
    trainingLabels = Mat(labels).reshape(1, labels.size());
    cv::Ptr<cv::face::LBPHFaceRecognizer> faceRecognizer;
    Ptr<KNearest> knn;
    if (algo == "knn")
    {
        knn = KNearest::create();
        knn->setDefaultK(3);
        knn->train(trainingData, ROW_SAMPLE, trainingLabels);
    }
    else
    {
        faceRecognizer = cv::face::LBPHFaceRecognizer::create();
        faceRecognizer->train(recognizerTrainingData, labels);
    }

    for (size_t video_index = 0; video_index < videos.size(); video_index++)
    {
        Mat frame;
        vid = videos[video_index];
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
                        cv::Ptr<cv::Tracker> tracker;
                        if (tr == "KCF")
                        {
                            tracker = cv::TrackerKCF::create();
                        }
                        else
                        {
                            tracker = cv::TrackerTLD::create();
                        }
                        cv::Rect myROI(faces[face].x, faces[face].y, faces[face].width, faces[face].height);
                        tracker->init(frame, myROI);
                        // Resize the ROI to match the HOG descriptor window size
                        trackers.push_back(tracker);
                        // imwrite(to_string(frameIndex) + "c.jpg", frame(myROI));
                        // waitKey();
                    }
                };
            };
            if (!faces.size())
                continue;
            for (size_t i = 0; i < faces.size(); i++)
            {
                cv::Rect2d myROI(faces[i].x, faces[i].y, faces[i].width, faces[i].height);
                bool success = trackers[i]->update(frame, myROI);
                ensureROIInBounds(myROI, frame.size());
                if (success)
                {
                    cv::Mat testImage = frame(myROI);
                    cvtColor(testImage, testImage, COLOR_BGR2GRAY);
                    std::vector<float> testDescriptors;
                    cv::resize(testImage, testImage, cv::Size(128, 256));
                    if (algo == "knn")
                    {
                        hog.compute(testImage, testDescriptors);
                        Mat testMat = Mat(testDescriptors).t();
                        cv::normalize(testMat, testMat, 0, 1, cv::NORM_MINMAX);
                        Mat results, neighborResponses, dists;
                        float response = knn->findNearest(testMat, 3, results, neighborResponses, dists);
                        float avgDistance = cv::mean(dists)[0];
                        cout << avgDistance << endl;
                        float distanceThreshold = 525;
                        if (avgDistance < distanceThreshold)
                        {
                            // Person is known
                            cv::putText(frame, labelsMap[response], cv::Point(faces[i].x, faces[i].y),
                                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
                            std::cout << "Person Found! Label: " << labelsMap[response] << std::endl;
                            cv::rectangle(frame, myROI, cv::Scalar(0, 255, 0), 2, 1);
                        }
                        else
                        {
                            // Person is unknown
                            std::cout << "Person Not Found or Unknown!" << std::endl;
                            cv::rectangle(frame, myROI, cv::Scalar(0, 0, 255), 2, 1);
                            cv::putText(frame, "Unknown", cv::Point(faces[i].x, faces[i].y),
                                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 2);
                        }
                    }
                    else
                    {
                        // Predict the label
                        int label;
                        double confidence;
                        faceRecognizer->predict(testImage, label, confidence);
                        if (confidence < 70)
                        {
                            // Person is known
                            cv::putText(frame, labelsMap[label], cv::Point(faces[i].x, faces[i].y),
                                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
                            std::cout << "Person Found! Label: " << labelsMap[label] << std::endl;
                            cv::rectangle(frame, myROI, cv::Scalar(0, 255, 0), 2, 1);
                        }
                        else
                        {
                            // Person is unknown
                            std::cout << "Person Not Found or Unknown!" << std::endl;
                            cv::rectangle(frame, myROI, cv::Scalar(0, 0, 255), 2, 1);
                            cv::putText(frame, "Unknown", cv::Point(faces[i].x, faces[i].y),
                                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 2);
                        }
                    }
                }

                else
                {
                    std::cerr << "Tracking failure for ROI " << i << std::endl;
                }
            }

            // cv::Rect myROI(10, 10,100, 100);
            show(video_name, frame);
        }
    }
    return 0;
}