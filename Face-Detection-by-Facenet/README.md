# Face Recogntion with Facenet
* This project aims to detect the face of the persons using facenet with tensorflow library.<br>
* The project includes improved image cropping and alignment using Multi-Task Cascaded Convolution Neural Network. In this project a 128-dimensional embedding based on triplet loss and deep CNN is used for classification. We have achieved an accuracy of 89%.<br>
* The project is implemented in Python.<br>
* The working video of the model is also in this repository with the file named as "Video of Facenet".<br>
* The working video of our model from facenet is [CLICK HERE](https://www.youtube.com/watch?v=6Pm2-S2MhMs "Click to Watch!") <br>
<h3> Multi-task Cascaded Convolutional Neural Network </h3>
<p> Pipeline of our cascaded framework that includes three-stage multi-task deep convolutional networks. Firstly, candidate windows are produced
through a fast Proposal Network (P-Net). After that, we refine these candidates
in the next stage through a Refinement Network (R-Net). In the third stage,
The Output Network (O-Net) produces final bounding box and facial landmarks position.
<p align="center">
<img src = "https://github.com/braghav968/Automated-Check-in-System/blob/master/Face-Detection-by-Facenet/MTCNN.jpg" height = 300>
</p>
<h3> Input Image
<p align="center">
<img src = "https://github.com/braghav968/Automated-Check-in-System/blob/master/Face-Detection-by-Facenet/images/Raghav/IMG_20170606_064101.jpg" height = 200>
</p>
</h3>
<h3> Aligned and Cropped Image using Multi-task Cascaded Convolutional Neural Networks
<p align="center">
<img src = "https://github.com/braghav968/Automated-Check-in-System/blob/master/Face-Detection-by-Facenet/aligned_images/Raghav/IMG_20170606_064101.png" height = 200>
</p>
</h3>
