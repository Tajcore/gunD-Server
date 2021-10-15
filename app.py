from flask import Flask, request, jsonify
from flask_cors import CORS
from bson.json_util import dumps
from json import loads
import numpy as np
import urllib
import cv2
import os

# Suppress tensorflow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

# Initialize flask app
app = Flask(__name__)
CORS(app)

# Initialize model
model = None

def load_model():
    global model
    model = tf.keras.models.load_model('detector.h5')

def url_to_image(url):
    # Open image, convert it to np array and read it into cv2
    resp = urllib.request.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image

def prepare_image(image, target):
    # Prepare the image for input to the model
    input_image = cv2.resize(image, target, interpolation=cv2.INTER_AREA)
    #print(input_image.shape)
    input_image = np.expand_dims(input_image, axis=0)
    return input_image/255.0

@app.route('/predict', methods=['POST'])
def predict():
    # Get the url to the image
    url = request.json['url']

    # Load the image from the URL and extract its dimensions
    image = url_to_image(url)
    #image = imutils.resize(image, width=600)
    (h, w) = image.shape[:2]

    # Prepare the image for input to the mode
    input_image = prepare_image(image, target=(224, 224))

    # Get preds from model
    preds = model.predict(input_image)[0]
    (startX, startY, endX, endY) = preds
    startX = int(startX * w)
    startY = int(startY * h)
    endX = int(endX * w)
    endY = int(endY * h)

    # Store the coordinates in a dictionary
    coords = {
        "x1": startX,
        "y1": startY,
        "x2": endX,
        "y2": endY
    }

    # cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
    # cv2.imshow('Image', image)
    # cv2.waitKey(0)

    return jsonify(loads(dumps(coords)))

if __name__ == '__main__':
    load_model()
    app.run(
        debug=True
    )


    