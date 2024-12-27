# Steps to reproduce


```bash
mkdir build

cd build

cmake ..

cmake --build .

./bin/runner
```

make sure you are running the exe file from `build` folder

prompts will help you run the code on all or different files with KNN or FaceRecognition

# Findings

The standart extractors and face finding algorithms lack precision and they are prone to underfiting.

Dataset of 10-13 images per person is not sufficient to guarantee precise recognition. In videos with different lightings, face positions and multiple faces of other people.

Preprocessing of the images of a person drastically helps, mirroring and blurring made the model more precise (approx. 2 times).

Switching from KFD to TLD tracking made model perform a lot worse, for some reason.

Tuning thresholds for KNN and FaceRecognition also improved performance a lot.

KNN tends to predict unknown faces as known, when they are facing the camera.

Faces with specific features (such as glasses) are better recognized, e.g. liev_schreiber
