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


The standard extractors and face-finding algorithms lack precision and are prone to underfitting.

A dataset of 10–13 images per person is insufficient to guarantee accurate recognition in videos with varying lighting conditions, face positions, and the presence of multiple faces from other individuals.

Preprocessing the images of a person significantly improves accuracy. Techniques such as mirroring and blurring enhanced the model's precision by approximately twofold.

Switching from KFD to TLD tracking significantly worsened the model's performance for an unknown reason.

Adjusting the thresholds for KNN and FaceRecognition also greatly improved performance.

KNN tends to misclassify unknown faces as known when the faces are directly facing the camera.

Faces with distinctive features, such as glasses, are recognized more reliably (e.g., liev_schreiber).
