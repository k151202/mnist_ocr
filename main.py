import tensorflow as tf
import numpy as np
import json
import cv2
from flask import Flask, render_template, request

app = Flask(__name__)

model = tf.keras.models.load_model("model")


def contour_rank(contour, cols):
    """determine sequence of contours"""
    tolerance_factor = 50
    origin = cv2.boundingRect(contour)
    return ((origin[1] // tolerance_factor) * tolerance_factor) * cols + origin[0]


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api", methods=["GET", "POST"])
def api():
    if request.method == "POST":
        # read file as bytes and decoding
        data = request.files["file"].read()
        raw = np.fromstring(data, dtype=np.uint8)
        img = cv2.imdecode(raw, cv2.IMREAD_COLOR)

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # generate binary image
        _, flag = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
        # inverse binary image for finding contours
        _, img_th = cv2.threshold(flag, 127, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(
            img_th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        contours = tuple(
            sorted(contours, key=lambda x: contour_rank(x, img_th.shape[1]))
        )
        rects = [cv2.boundingRect(contour) for contour in contours]

        img_for_class = flag.copy()
        mnist_imgs = []
        margin = 12

        for rect in rects:
            im = img_for_class[
                rect[1] - margin : rect[1] + rect[3] + margin,
                rect[0] - margin : rect[0] + rect[2] + margin,
            ]
            row, col = im.shape[:2]

            border_size = max(row, col)
            diff = min(row, col)

            bottom = im[row - 2 : row, 0:col]
            mean = cv2.mean(bottom)[0]

            border = cv2.copyMakeBorder(
                im,
                top=0,
                bottom=0,
                left=int((border_size - diff) / 2),
                right=int((border_size - diff) / 2),
                borderType=cv2.BORDER_CONSTANT,
                value=[mean, mean, mean],
            )

            resized_img = cv2.resize(
                border, dsize=(28, 28), interpolation=cv2.INTER_LINEAR
            )
            mnist_imgs.append(resized_img)

        results = []

        for i in range(len(mnist_imgs)):
            img = mnist_imgs[i].reshape(-1, 28, 28, 1)
            input_data = ((np.array(img) / 255.0) - 1) * -1
            answer = int(np.argmax(model.predict(input_data), axis=-1))
            results.append({"num": i + 1, "answer": answer})

    return json.dumps(results)


if __name__ == "__main__":
    # app.run(host='0.0.0.0', port=8080)
    app.run(debug=True)
