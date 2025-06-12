from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import uuid
import os
from collections import Counter

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

PIXELS_PER_CM = 37.8 

def classify_shape(approx):
    sides = len(approx)
    if sides <= 5:
        return "hook_shape"
    elif 6 <= sides <= 14:
        return "multi-slot block"
    else:
        return "rod"

def calculate_dimensions(contour):
    x, y, w, h = cv2.boundingRect(contour)
    width_cm = round(w / PIXELS_PER_CM, 2)
    height_cm = round(h / PIXELS_PER_CM, 2)
    return width_cm, height_cm, x, y, w, h

def is_significantly_different(box1, box2, threshold=10):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    return (
        abs(x1 - x2) > threshold or
        abs(y1 - y2) > threshold or
        abs(w1 - w2) > threshold or
        abs(h1 - h2) > threshold
    )

@app.get("/", response_class=HTMLResponse)
async def form():
    return """
    <html>
        <head><title>Object Dimension Detector</title></head>
        <body>
            <h2>Upload Image to Detect Objects and Measure in cm</h2>
            <form action="/analyze/" enctype="multipart/form-data" method="post">
                <input type="file" name="file" accept="image/*" required>
                <input type="submit" value="Analyze">
            </form>
        </body>
    </html>
    """

@app.post("/analyze/", response_class=HTMLResponse)
async def analyze(file: UploadFile = File(...)):
    temp_file = f"temp_{uuid.uuid4()}.jpg"
    with open(temp_file, "wb") as f:
        f.write(await file.read())

    image = cv2.imread(temp_file)
    os.remove(temp_file)

    if image is None:
        return HTMLResponse("<h3>Invalid image file.</h3>", status_code=400)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    edged = cv2.Canny(blurred, 20, 100)
    edged = cv2.dilate(edged, None, iterations=1)

    # Only external contours
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    unique_boxes = []
    detected = []
    shape_names = []

    for contour in contours:
        if cv2.contourArea(contour) < 300:
            continue

        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        width_cm, height_cm, x, y, w, h = calculate_dimensions(contour)

        current_box = (x, y, w, h)
        is_duplicate = False

        for box in unique_boxes:
            if not is_significantly_different(current_box, box):
                is_duplicate = True
                break

        if not is_duplicate:
            shape = classify_shape(approx)
            detected.append({
                "shape": shape,
                "width_cm": width_cm,
                "height_cm": height_cm
            })
            unique_boxes.append(current_box)
            shape_names.append(shape)

    type_count = dict(Counter(shape_names))
    html = f"<h2>Total Objects Detected: {len(detected)}</h2>"
    html += "<h3>Type-wise Count:</h3><ul>"
    for k, v in type_count.items():
        html += f"<li>{k}: {v}</li>"
    html += "</ul>"

    html += "<h3>Object Details:</h3><ol>"
    for obj in detected:
        html += f"<li>Shape: {obj['shape']}, Width: {obj['width_cm']} cm, Height: {obj['height_cm']} cm</li>"
    html += "</ol><a href='/'>← Upload another image</a>"

    return HTMLResponse(html)
