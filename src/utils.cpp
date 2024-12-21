#include <string>
#include <iostream>
#include "utils.h"

using namespace std;

string video_path(string path)
{
    cout << path << endl;
    return "./videos/" + path;
}