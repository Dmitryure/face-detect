#include "detectors.h"
#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"

using namespace cv;

std::vector<Rect> detectAndDisplay(Mat frame)
{
    cv::CascadeClassifier face_cascade("haarcascade_frontalface_alt.xml");
    cv::CascadeClassifier eyes_cascade("haarcascade_eye_tree_eyeglasses.xml");
    equalizeHist(frame, frame);
    //-- Detect faces
    std::vector<Rect> faces;
    face_cascade.detectMultiScale(frame, faces);
    for (size_t i = 0; i < faces.size(); i++)
    {
        Point center(faces[i].x + faces[i].width / 2, faces[i].y + faces[i].height / 2);
        Mat faceROI = frame(faces[i]);
        //-- In each face, detect eyes
        std::vector<Rect> eyes;
        eyes_cascade.detectMultiScale(faceROI, eyes);
        for (size_t j = 0; j < eyes.size(); j++)
        {
            Point eye_center(faces[i].x + eyes[j].x + eyes[j].width / 2, faces[i].y + eyes[j].y + eyes[j].height / 2);
            int radius = cvRound((eyes[j].width + eyes[j].height) * 0.25);
        }
    }
    return faces;
}