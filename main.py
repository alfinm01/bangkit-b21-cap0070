import io
import json
import base64
import requests
import numpy as np
import pandas as pd
from PIL import Image
from flask_cors import CORS
from flask import Flask, request
from keras.models import load_model
from keras.preprocessing.image import img_to_array

API_KEY = "AIzaSyCRUguLezAfzQrLM3TiWcQCWIE9q0eiaHc"

app = Flask(__name__)
CORS(app)

def load_model_global():
    print("load_model_global started...")
    global model
    model = load_model("./model/nutrition-analyzer-model.h5")
    print("load_model_global finished...")

def prepare_image(image, target):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = np.vstack([image])
    return image

def load_data():
    print("load_data started...")
    global nutrition_values
    url = './data/nutrition_values.csv'
    nutrition_values = pd.read_csv(url)
    nutrition_values["product"] = nutrition_values["product_name"].replace(" ", "_", regex=True)
    print("load_data finished...")

def get_food_data_from_index(indexes):
    top3_prediction = []
    for i, row in nutrition_values.iterrows():
        if i in indexes:
            nutrition = {
                "name": row["product_name"],
                "calories": row["energy_100g"],
                "proteins": row["proteins_100g"],
                "fats": row["fat_100g"],
                "carbohydrates": row["carbohydrates_100g"],
                "sugars": row["sugars_100g"],
                "fibers": row["fiber_100g"],
                "cholesterols": row["cholesterol_100g"]
            }
            top3_prediction.append(nutrition)
    return top3_prediction

@app.route("/")
def health_check():
    return json.dumps({ "status_code": 200, "message": "ok" })

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return json.dumps({ "status_code": 400, "message": "please provide an image file" }), 400
    image = request.files["image"].read()
    image = Image.open(io.BytesIO(image))
    image = prepare_image(image, target=(150, 150))
    preds = model.predict(image, batch_size=10)
    y_pred_index = np.argsort(preds[0])[-3:]
    set_y_pred_index = set(y_pred_index)
    print("\nTop 3 prediction results (index):", y_pred_index)
    result = get_food_data_from_index(y_pred_index)
    print("\nTop 3 prediction results:", result)
    return json.dumps({ "status_code": 200, "data": result })

@app.route("/detect-text", methods=["POST"])
def detect_text():
    if "image" not in request.files:
        return json.dumps({ "status_code": 400, "message": "please provide an image file" }), 400
    image = request.files["image"].read()
    base64_bytes = base64.b64encode(image)
    base64_string = base64_bytes.decode('utf-8')
    payload = {
        "requests": [
            {
                "image": {
                    "content": base64_string
                },
                "features": [
                    {
                        "type": "TEXT_DETECTION"
                    }
                ]
            }
        ]
    }
    response = requests.post("https://vision.googleapis.com/v1/images:annotate?key=" + API_KEY, data=json.dumps(payload, indent=2)).json()
    if "responses" in response and "textAnnotations" in response["responses"][0]:
        return json.dumps({ "status_code": 200, "data": response["responses"][0]["textAnnotations"][0] })
    return json.dumps({ "status_code": 200, "message": "did not found specific annotation", "data": response })

@app.route("/detect-image", methods=["POST"])
def detect_image():
    if "image" not in request.files:
        return json.dumps({ "status_code": 400, "message": "please provide an image file" }), 400
    image = request.files["image"].read()
    base64_bytes = base64.b64encode(image)
    base64_string = base64_bytes.decode('utf-8')
    payload = {
        "requests": [
            {
                "image": {
                    "content": base64_string
                },
                "features": [
                    {
                        "type": "WEB_DETECTION"
                    }
                ]
            }
        ]
    }
    response = requests.post("https://vision.googleapis.com/v1/images:annotate?key=" + API_KEY, data=json.dumps(payload, indent=2)).json()
    return json.dumps({ "status_code": 200, "data": response })

if __name__ == '__main__':
    print(("* Loading Keras model and Flask starting server..."
        "please wait until server has fully started"))
    load_model_global()
    load_data()
    app.run(host="0.0.0.0")