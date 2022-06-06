# Imports
import json
import werkzeug
from flask import Flask, request, jsonify
import pickle

from main import show_image_test

# Flask
app = Flask(__name__)

# Routes for API

# /emotion_detection POST Route detects the emotion using show_image_test function in main.py and returns the list of faces coordinate
# in the body of the request we need to pass the image.

@app.route("/emotion_detection", methods=["POST"])
def index():
    # Image
    imagefile = request.files["image"]
    # Getting file name of the image using werkzeug library
    filename = werkzeug.utils.secure_filename(imagefile.filename)
    print("\nReceived image File name : " + imagefile.filename)
    # Saving the image in images Directory
    imagefile.save("./images/" + filename)
    # Passing the imagePath in this show_image_test function and get emotion
    #answer = show_image_test("D:\COLLEGE\S8\Project\EMOTION DETECTION FINAL\images\" + filename)
    
    emotions = ["Angry","Happy","Sad"]
    pkl_file = open('./model.pkl', 'rb')
    data = pickle.load(pkl_file)
    pkl_file.close()
    #d = {}
    answer = show_image_test(data, emotions, "./images/" + filename)
    #d['emotion'] = str(answer)
    return json.dumps({"emotion": answer})

    # Returns faces Cordinate in the json Format
    #return json.dumps({"faces": faces})

# Running the app
app.run()