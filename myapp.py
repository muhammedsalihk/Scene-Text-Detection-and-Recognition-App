import streamlit as st
import cv2
from PIL import Image,ImageEnhance
from imutils.object_detection import non_max_suppression
import numpy as np
import argparse
import time
import cv2
import math
import pytesseract
import tempfile
from matplotlib.pyplot import imshow

def process_image_for_ocr(img_crop):
    im_new = cv2.GaussianBlur(img_crop,(5,5),0)
    return im_new

def set_image_dpi(img_crop):
    IMAGE_SIZE = 1800
    im = img_crop
    length_x, width_y = im.size
    print(im.size)
    factor = max(1, int(IMAGE_SIZE / length_x))
    size = factor * length_x, factor * width_y
    print(size)
    im_resized = im.resize(size, Image.ANTIALIAS)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
    temp_filename = temp_file.name
    im_resized.save(temp_filename, dpi=(300, 300))
    return temp_filename

def image_smoothening(img):
    BINARY_THREHOLD = 180
    ret1, th1 = cv2.threshold(img, BINARY_THREHOLD, 255, cv2.THRESH_BINARY)
    ret2, th2 = cv2.threshold(th1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    blur = cv2.GaussianBlur(th2, (1, 1), 0)
    ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th3

def remove_noise_and_smooth(file_name):
    img = cv2.imread(file_name, 0)
    filtered = cv2.adaptiveThreshold(img.astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 41,
                                     3)
    kernel = np.ones((1, 1), np.uint8)
    opening = cv2.morphologyEx(filtered, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    img = image_smoothening(img)
    or_image = cv2.bitwise_or(img, closing)
    return or_image


def detectword(word_img):
    config = ('-l eng --oem 1 --psm 8')
    #processed = process_image_for_ocr(word_img)
    processed = word_img
    text = pytesseract.image_to_string(processed, config=config, timeout=2)
    return text


def decode(scores, geometry, scoreThresh):
    
    (numRows, numCols) = scores.shape[2:4]
    detections = []
    confidences = []
    # loop over the number of rows
    for y in range(0, numRows):
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]
        for x in range(0, numCols):
            if scoresData[x] < scoreThresh:
                continue
            (offsetX, offsetY) = (x * 4.0, y * 4.0)
            angle = anglesData[x]
            cosA = math.cos(angle)
            sinA = math.sin(angle)
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]
            # Calculate offset
            offset = ([offsetX + cosA * xData1[x] + sinA * xData2[x], offsetY - sinA * xData1[x] + cosA * xData2[x]])
            # Find points for rectangle
            p1 = (-sinA * h + offset[0], -cosA * h + offset[1])
            p3 = (-cosA * w + offset[0],  sinA * w + offset[1])
            center = (0.5*(p1[0]+p3[0]), 0.5*(p1[1]+p3[1]))
            detections.append((center, (w,h), -1*angle * 180.0 / math.pi))
            confidences.append(float(scoresData[x]))

    return [detections, confidences]


def text_detect(img):
    new_img = np.array(img.convert('RGB'))
    #new_img = np.array(img)
    
    image = cv2.cvtColor(new_img,1)
    orig = image.copy()
    text_img = img

    (H, W) = image.shape[:2]

    newW = round(W/32) * 32
    newH = round(H/32) * 32

    rW = W / float(newW)
    rH = H / float(newH)

    net = cv2.dnn.readNet('frozen_east_text_detection.pb')
    layerNames = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]

    blob = cv2.dnn.blobFromImage(image, 1.0, (newW, newH), (123.68, 116.78, 103.94), swapRB=True, crop=False)
    net.setInput(blob)
    (scores, geometry) = net.forward(layerNames)

    confThreshold = 0.5
    nmsThreshold = 0.4

    [boxes, confidences] = decode(scores, geometry, confThreshold)
    indices = cv2.dnn.NMSBoxesRotated(boxes, confidences, confThreshold, nmsThreshold)
    
    
    
    for i in indices:
        # get 4 corners of the rotated rect
        vertices = cv2.boxPoints(boxes[i[0]])
        # scale the bounding box coordinates based on the respective ratios
        for j in range(4):
            vertices[j][0] *= rW
            vertices[j][1] *= rH
       
        ymax = max(vertices[:,0])
        ymin = min(vertices[:,0])
        xmax = max(vertices[:,1])
        xmin = min(vertices[:,1])

        pad = 0.005
        
        height = (xmax - xmin)
        dY = (ymax - ymin) * pad
        dX = (xmax - xmin) * pad 

        ymax = int(min(W, ymax + dY))
        xmax = int(min(H, xmax + dX))

        ymin = int(max(0, ymin - dY))
        xmin = int(max(0, xmin - dX))

        if height > (0.035*H):
            cropped = text_img.crop((ymin, xmin, ymax, xmax))
            #cropped = orig[xmin:xmax, ymin:ymax]
            text = detectword(cropped)
        
        else:
            text =''

        for j in range(4):
            p1 = (vertices[j][0], vertices[j][1])
            p2 = (vertices[(j + 1) % 4][0], vertices[(j + 1) % 4][1])
            cv2.line(orig, p1, p2, (0, 255, 0), 2, cv2.LINE_AA)
        
        if (text != ''):
            cv2.putText(orig, text, (vertices[1][0], vertices[1][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    return orig


def main():
    st.title('Text Detection App')
    st.text('Built by Muhammed Salih')

    st.subheader('From Image')
    image_file = st.file_uploader('Upload Image', type=['jpg', 'jpeg', 'png'])

    if image_file is not None:
        img = Image.open(image_file)
        st.text('Uploaded Image')
        st.image(img, use_column_width=True)

        if st.button('Detect Text'):
            detect = text_detect(img)
            st.text('Image with detected texts')
            cv2.destroyAllWindows()
            st.image(detect, use_column_width=True)       


if __name__ == '__main__':
		main()
