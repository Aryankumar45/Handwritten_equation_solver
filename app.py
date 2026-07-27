from flask import Flask, render_template, request
from preprocess import preprocess_equation
from model.predict import get_equation
from solve import solve_eq
import base64
import cv2
import os

os.system("cls")
app = Flask(__name__)

TEMP_IMAGE = "static/temp.png"

@app.route("/", methods=["GET", "POST"])
def home():

    if request.method == "GET":
        return render_template("index.html", image_data=None, equation=None, result=None)

    action = request.form.get("action")

    if action == "preview":
        image = request.files.get("image")
        if image is None or image.filename == "":
            return render_template("index.html", image_data=None, equation=None, result="Please select an image.")

        image_bytes = image.read()
        with open(TEMP_IMAGE, "wb") as f:
            f.write(image_bytes)

        image_data = base64.b64encode(image_bytes).decode("utf-8")
        with open(TEMP_IMAGE, "rb") as f:
            thresh, _ = preprocess_equation(f)

        preview = cv2.bitwise_not(thresh)
        if len(preview.shape) == 2:
            preview = cv2.cvtColor(preview, cv2.COLOR_GRAY2BGR)

        _, buffer = cv2.imencode(".png", preview)
        thresh_data = base64.b64encode(buffer).decode("utf-8")
        return render_template("index.html", image_data=image_data, thresh_data=thresh_data, equation=None, result=None)

    elif action == "solve":
        if not os.path.exists(TEMP_IMAGE):
            return render_template("index.html", image_data=None, equation=None, result="Please upload an image first.")

        with open(TEMP_IMAGE, "rb") as f:
            image_bytes = f.read()

        image_data = base64.b64encode(image_bytes).decode("utf-8")
        with open(TEMP_IMAGE, "rb") as f:
            thresh, characters = preprocess_equation(f)

        equation = get_equation(characters)
        result = solve_eq(equation)
        return render_template("index.html", image_data=image_data, equation=equation, result=result)

    elif action == "sample_preview":
        filename = request.form.get("sample_image")
        if filename is None:
            return render_template("index.html", image_data=None, equation=None, result="Please select a sample image.")

        path = os.path.join("static", "sample", filename)
        with open(path, "rb") as f:
            image_bytes = f.read()

        with open(TEMP_IMAGE, "wb") as f:
            f.write(image_bytes)

        image_data = base64.b64encode(image_bytes).decode("utf-8")
        with open(TEMP_IMAGE, "rb") as f:
            thresh, _ = preprocess_equation(f)

        preview = cv2.bitwise_not(thresh)
        if len(preview.shape) == 2:
            preview = cv2.cvtColor(preview, cv2.COLOR_GRAY2BGR)

        _, buffer = cv2.imencode(".png", preview)
        thresh_data = base64.b64encode(buffer).decode("utf-8")
        return render_template("index.html", image_data=image_data, thresh_data=thresh_data, equation=None, result=None)

    return render_template("index.html", image_data=None, equation=None, result="Invalid request.")

if __name__ == "__main__":
    app.run(debug=True)