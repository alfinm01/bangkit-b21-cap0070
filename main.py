import json
import base64
import requests
from flask import Flask, request

API_KEY = "AIzaSyCRUguLezAfzQrLM3TiWcQCWIE9q0eiaHc"
app = Flask(__name__)

@app.route("/")
def health_check():
    return json.dumps({ "status_code": 200, "message": "ok" })

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
    if response["responses"] and response["responses"][0]["textAnnotations"]:
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
    app.run(host="0.0.0.0")
    # app.run(debug=True)