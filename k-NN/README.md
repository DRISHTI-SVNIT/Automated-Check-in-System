To use this you must have following packages installed in your python environment,other than OpenCV
  -dlib
  -face_recognition
  -imutils

Linux Users can install these packages using pip command as follows,
pip install dlib
pip install face_recognition
pip install imutils

Next, Create a folder in dataset folder with the name of the person.
Add photos of that person in that folder, take care the folder name is same as person's name, photos name can be anything.
Similarly, create folder for each person you want to have your model work for.

Next, Run encode_faces.py script with two command-line arguments providing location of the dataset and location of the file where encodings are to be stored, as follows
python encode_faces.py --dataset dataset --encodings encodings.pickle

Then, Run recognize_faces.py script with two command-line arguments, stating the encodings file and path and name of the output video (second argument is optinonal), as follows,
python recognize_faces.py --encodings encodings.pickle --output output/trial.avi

It is capable enough of recognizing all trained faces in single frame.

A sample output video and dataset is included in output and dataset folder respectively.

