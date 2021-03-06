import face_recognition
import os
import cv2
import dlib
import dlib.cuda as cuda

print(dlib.DLIB_USE_CUDA)
print(cuda.get_num_devices())

KNOWN_FACES_DIR = 'known_faces'
TOLERANCE = 0.5
FRAME_THICKNESS = 3
FONT_THICKNESS = 2
MODEL = "cnn" # or hog

video = cv2.VideoCapture(0, cv2.CAP_DSHOW)

def testDevice(source):
   cap = cv2.VideoCapture(source) 
   if cap is None or not cap.isOpened():
       print('Warning: unable to open video source: ', source)

testDevice(0) # no printout
testDevice(1) # prints message

def name_to_color(name):
    # Take 3 first letters, tolower()
    # lowercased character ord() value rage is 97 to 122, substract 97, multiply by 8
    color = [(ord(c.lower())-97)*8 for c in name[:3]]
    return color

print("loading known faces...")

known_faces = []
known_names = []

# Listing and appending each image and directory to known faces, names
for name in os.listdir(KNOWN_FACES_DIR):
	for filename in os.listdir(f"{KNOWN_FACES_DIR}/{name}"):
		image = cv2.imread(f"{KNOWN_FACES_DIR}/{name}/{filename}")
		if (image.shape[0] > 800 or image.shape[1] > 500):
			scale_percent = 50
			width = int(image.shape[1] * scale_percent / 100)
			height = int(image.shape[0] * scale_percent / 100)
			dim = (width, height)
			image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
		if len(face_recognition.face_encodings(image)) > 0:
			encoding = face_recognition.face_encodings(image)[0]
			known_faces.append(encoding)
			known_names.append(name)

# for each unknown face, we would want to find the face and then classify as either me or natsuko
while True:
	ret, image = video.read()

	# this is the facial recognition part - finds where the found faces are
	locations = face_recognition.face_locations(image, number_of_times_to_upsample=0, model = MODEL)
	encodings = face_recognition.face_encodings(image, locations)

	for face_encoding, face_location in zip(encodings, locations):
		results = face_recognition.compare_faces(known_faces, face_encoding, TOLERANCE)
		match = None
		if True in results:
			match = known_names[results.index(True)]
			print(f"Match found: {match}")

			top_left = (face_location[3], face_location[0])
			bottom_right = (face_location[1], face_location[2])
			color = name_to_color(match)
			cv2.rectangle(image, top_left, bottom_right, color, FRAME_THICKNESS)

			top_left = (face_location[3], face_location[2])
			bottom_right = (face_location[1], face_location[2] + 22)
			cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)
			cv2.putText(image, match, (face_location[3] + 10, face_location[2] + 15), \
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), FONT_THICKNESS)

	cv2.imshow(filename, image)
	if cv2.waitKey(1) & 0xFF == ord("q"):
		break