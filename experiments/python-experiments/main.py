# import the necessary packages
from imutils import paths
import numpy as np
import imutils
import cv2
import os

# setting the arguments
args = {}
currentdir = os.getcwd()
args["detector"] = currentdir + '/face_detection_model'
args["embedding_model"] = currentdir + '/openface_nn4.small2.v1.t7'
args["embeddings"] = currentdir + '/output/embeddings.pickle'
#args["dataset"] = '/run/user/1000/gvfs/smb-share:server=nas-01.kb.dk,share=faelles2/IT/ITU/daner/portraits'
#args["destination"] = '/run/user/1000/gvfs/smb-share:server=nas-01.kb.dk,share=faelles2/IT/ITU/daner/face2/'
args["dataset"] = currentdir + '/test1'
args["destination"] = currentdir + '/test2/'
args["confidence"] = 0.5

# load our serialized face detector from disk
print("[INFO] loading face detector...")
protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
modelPath = os.path.sep.join([args["detector"], "res10_300x300_ssd_iter_140000.caffemodel"])
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# load our serialized face embedding model from disk
print("[INFO] loading face recognizer...")
embedder = cv2.dnn.readNetFromTorch(args["embedding_model"])

# grab the paths to the input images in our dataset
print("[INFO] quantifying faces...")
imagePaths = list(paths.list_images(args["dataset"]))

# loop over the image paths
for (i, imagePath) in enumerate(imagePaths):
    # extract the image name from the image path
    # load the image, resize it to have a width of 600 pixels (while
    # maintaining the aspect ratio), and then grab the image
    # dimensions
    name = imagePath.split(os.path.sep)[-1]
    destination = args["destination"] + name
    print(name)
    image = cv2.imread(imagePath)
    if image.any():
        image = imutils.resize(image, width=600)
        (h, w) = image.shape[:2]
        # construct a blob from the image
        imageBlob = cv2.dnn.blobFromImage(
            cv2.resize(image, (300, 300)), 1.0, (300, 300),
            (104.0, 177.0, 123.0), swapRB=False, crop=False)
        # apply OpenCV's deep learning-based face detector to localize
        # faces in the input image
        detector.setInput(imageBlob)
        detections = detector.forward()
        # ensure at least one face was found
        if len(detections) > 0:
            # we're making the assumption that each image has only ONE
            # face, so find the bounding box with the largest probability
            i = np.argmax(detections[0, 0, :, 2])
            confidence = detections[0, 0, i, 2]
            # ensure that the detection with the largest probability also
            # means our minimum probability test (thus helping filter out
            # weak detections)
            if confidence > args["confidence"]:
                # compute the (x, y)-coordinates of the bounding box for
                # the face
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                # extract the face ROI and grab the ROI dimensions
                face = image[startY:endY, startX:endX]
                (fH, fW) = face.shape[:2]
                # ensure the face width and height are sufficiently large
                if fW < 20 or fH < 20:
                    continue

                # change the width and height to 70% of the found face and make it square
                Y70 = int((endY - startY) * .2)
                X70 = int((((endY + Y70) - (startY - Y70)) - (endX - startX)) / 2)
                face = image[startY - Y70:endY + Y70, startX - X70:endX + X70]

                # change the aspect ratio to 256 * 256
                face = cv2.resize(face, (256, 256))
                cv2.imwrite(destination, face)
