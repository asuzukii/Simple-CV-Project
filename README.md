# Simple-CV-Project

This is a small project I did in order to properly classify faces via cv2 and face_recognition.

The first file, RecognizeFaces.py, is a static classifier that takes in a small sample of already labelled images (or more precisely I put images of the same person in corresponding folders) and trains on those pictures on top of the pre-trained imagenet model that the face_recognition import already has. Then, I can feed in pictures with unknown labels in order for the program to detect then classify the faces in the picture and determine if the face found was one of the people listed in the known faces list, or a completely new person.

In the second file, RecognizeVideo.py, we use live video feed in order to do the same process that we did above, just substituting inputted pictures of unlabelled people with live camera action. Although the recognition does work in principal, in practical terms, the feed takes too long to process so the camera feedback runs at ~0.5 fps. This is either from the lack of equipment that I have at my house (is GTX 1660 really gonna not cut it for this program??) or more likely, the dlib library I imported that is supposed to help massively in run time is not working as intended.

For future updates, I'd like to first improve the performance time on the video feed as well as the static processing, but more in the long run, I'd be able to correctly find new faces in live video feed and label them as well so that I could use the same algorithm on an auto-human detection camera system with raspberry pi.
