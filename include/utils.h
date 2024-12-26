#include <string>

using namespace std;

string video_path(string path);
map<string, vector<cv::Mat>> load_images();
void show(string window_name, cv::Mat img);
std::vector<cv::VideoCapture> load_videos();