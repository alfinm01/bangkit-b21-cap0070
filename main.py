from flask import Flask, request

app = Flask(__name__)

@app.route("/")
def health_check():
    return { "status_code": 200, "message": "ok" }

if __name__ == '__main__':
    app.run()