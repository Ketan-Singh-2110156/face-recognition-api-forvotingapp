import os
import cv2
import numpy as np
from pymongo import MongoClient
from flask import Flask, request, jsonify
import requests
from PIL import Image
from facenet_pytorch import InceptionResnetV1
from torchvision import transforms

uri = "mongodb+srv://simranyadav464:f0Kr8TnpOsc9x41F@voting-app.5v0a1bg.mongodb.net/?retryWrites=true&w=majority"
client = MongoClient(uri)

test_db = client.test
user_data = test_db.users

def find_by_adhaar(aadhar_no):
    people = user_data.find_one({"AadharNumber": aadhar_no})
    return people["userImage"]

app = Flask(__name__)

resnet = InceptionResnetV1(pretrained='vggface2').eval()

@app.route('/verify', methods=['POST'])
def verify_faces():
    try:
        image1 = request.files['image1']
        url = find_by_adhaar(request.form['aadhar_no'])
        img1_pil = Image.open(image1)
        img2_pil = Image.open(requests.get(url, stream=True).raw)

        if img1_pil is not None and img2_pil is not None:
            img1_tensor = transforms.functional.to_tensor(img1_pil).unsqueeze(0)
            img2_tensor = transforms.functional.to_tensor(img2_pil).unsqueeze(0)

            img1_embedding = resnet(img1_tensor)
            img2_embedding = resnet(img2_tensor)

            distance = np.linalg.norm(np.array(img1_embedding.detach().numpy()) - np.array(img2_embedding.detach().numpy()))

            threshold = 0.7

            if distance < threshold:
                result = "These faces are the same."
            else:
                result = "These faces are different."
        else:
            result = "No face detected in one or both images."

        return jsonify({"result": result})
        
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
