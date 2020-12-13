import sys
from joblib import dump, load
from sklearn.preprocessing import LabelEncoder, Normalizer
from sklearn.svm import SVC
import os
import cv2
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from keras.models import load_model
import numpy as np
import mtcnn
from mtcnn.mtcnn import MTCNN
from picamera import PiCamera
from picamera.array import PiRGBArray
import time

#https://machinelearningmastery.com/how-to-develop-a-face-recognition-system-using-facenet-in-keras-and-an-svm-classifier/
def extract_face(detector, pixels, required_size=(160, 160)):
  # detect faces in the image
  results = detector.detect_faces(pixels)
  
  if len(results) == 0:
      return None
  # extract the bounding box from the first face
  x1, y1, width, height = results[0]['box']
  # bug fix
  x1, y1 = abs(x1), abs(y1)
  x2, y2 = x1 + width, y1 + height
  # extract the face
  face = pixels[y1:y2, x1:x2]
  # resize pixels to the model size
  image = Image.fromarray(face)
  image = image.resize(required_size)
  face_array = np.asarray(image)
  return face_array

def load_image(detector, pixels, label):
  X, y = list(), list()
  
  faces = extract_face(detector, pixels)
  if faces is None:
      return None, None
  
  # create labels
  labels = [label]
  # store
  X.extend([faces])
  y.extend(labels)
  return np.asarray(X), np.asarray(y)

def get_embedding(model, face_pixels):
  # scale pixel values
  face_pixels = face_pixels.astype('float32')
  # standardize pixel values across channels (global)
  mean, std = face_pixels.mean(), face_pixels.std()
  face_pixels = (face_pixels - mean) / std
  # transform face into one sample
  samples = np.expand_dims(face_pixels, axis=0)
  # make prediction to get embedding
  yhat = model.predict(samples)
  return yhat[0]

def print_time_elapsed(starttime, message="Default"):
    endtime=time.time()
    diff = endtime - starttime
    print(message + " - Elapsed: " + str(diff))


def rectangle_overlay(o, message):
    camera.remove_overlay(o)
    img = img_empty.copy()
    draw = ImageDraw.Draw(img)
    draw.text((50,50), message, (255, 255, 255, 255))
    o = camera.add_overlay(img.tobytes(), layer=3, size=img.size, alpha=32)
    return o

face_detection_model = load_model('facenet_keras.h5')
face_classification_model = load('face_classifier.joblib')

out_encoder = LabelEncoder()
out_encoder.classes_ = np.load('classes.npy')
in_encoder = Normalizer(norm='l2')

detector = MTCNN()

camera = PiCamera(framerate=2, resolution=(736,480))
raw_capture = PiRGBArray(camera, size = camera.resolution)

time.sleep(0.5)

encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
win_xstart, win_ystart = 100, 100
win_width, win_height = 736, 480
camera.start_preview(fullscreen=False, window=(win_xstart,win_ystart,win_width,win_height))

img_empty = Image.new("RGBA", (736, 480))
img = Image.new("RGBA", (736, 480))
o = camera.add_overlay(img.tobytes(), layer=3, size=img.size, alpha=32)
# how to add multi-area text-overlay in picamera preview
#o = rectangle_overlay(o, "test")

for i in range(100):
    camera.capture(raw_capture, 'bgr')
    start=time.time()
    image_frame = raw_capture.array
    image_array = np.ascontiguousarray( image_frame[:,:,::-1], dtype=np.uint8)
    image, _ = load_image(detector, image_array, 'unknown')
    
    if image is not None:
        embedding = np.array([get_embedding(face_detection_model, image[0])])
#        print_time_elapsed(start, "Embedding")
        norm_embedding = in_encoder.transform(embedding)
        random_face_pixels = image[0]
        random_face_emb = norm_embedding[0]
        
        # prediction for the face
        samples = np.expand_dims(random_face_emb, axis=0)
        yhat_class = face_classification_model.predict(samples)
        yhat_prob = face_classification_model.predict_proba(samples)
        # get name
        class_index = yhat_class[0]
        class_probability = yhat_prob[0,class_index] * 100
        print_time_elapsed(start, "Predictions")

        predict_names = out_encoder.inverse_transform(yhat_class)
        if class_probability < 60:
          predict_names[0] = 'unknown'
        print('Predicted: %s (%.3f)' % (predict_names[0], class_probability))
        
        o = rectangle_overlay(o, str(predict_names[0]))
        print_time_elapsed(start, "Person")
    else:
        print("No person in the image")
        print_time_elapsed(start, "No person")
        print("----------------")
    raw_capture.truncate(0)
    time.sleep(0.1)
camera.remove_overlay(o)
camera.stop_preview()