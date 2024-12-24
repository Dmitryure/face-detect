#include <string>

using namespace std;

string video_path(string path);
vector<cv::Mat> load_images(string name);
void show(string window_name, cv::Mat img);