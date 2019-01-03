# imports packages
from imutils import paths
import face_recognition
import argparse
import pickle
import cv2
import os

# parses arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--dataset", required=True, help="path to input directory of faces + images")
ap.add_argument("-e", "--encodings", required=True, help="path to serialized db of facial encodings")
ap.add_argument("-d", "--detection-method", type=str, default="hog", help="face detection model to use: either `hog` or `cnn`")
args = vars(ap.parse_args())

# extracts paths to the input images
print("[INFO] quantifying faces...")
imagePaths = list(paths.list_images(args["dataset"]))

# initializes the list of known encodings and known names
knownEncodings = []
knownNames = []

# iterates over the image paths
for (i, imagePath) in enumerate(imagePaths):

	# extracts the name of the person from the image path
	print("[INFO] processing image {}/{}".format(i + 1, len(imagePaths)))
	name = imagePath.split(os.path.sep)[-2]

	image = cv2.imread(imagePath)
	rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

	# detects faces
	boxes = face_recognition.face_locations(rgb, model=args["detection_method"])

	# computes the facial embedding for the face
	encodings = face_recognition.face_encodings(rgb, boxes)

	# loops over the encodings
	for encoding in encodings:

		# adds each encoding + name to our set of known names and encodings
		knownEncodings.append(encoding)
		knownNames.append(name)

# writes the facial encodings + names to disk
print("[INFO] serializing encodings...")
data = {"encodings": knownEncodings, "names": knownNames}
f = open(args["encodings"], "wb")
f.write(pickle.dumps(data))
f.close()
