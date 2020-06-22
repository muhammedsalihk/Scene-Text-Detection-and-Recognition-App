# Scene Text Detection and Recognition App

## Introduction
Scene text recognition or the detection and recognition of text in natural images has a lot of applications in today’s world starting from areas like video analytics and robotics. In this project we are building, an interactive webapp using streamlit where an image can be uploaded and the result image with the recognised text and bounding boxes can be generated.

## Methodology
The first step of the process is localising or detecting the text in the images. We use a pretrained model implemented using Open CV’s dnn module. It returns the bounding boxes for the detected text areas. 

The text areas are extracted and preprocessed and then sent through an OCR module (we used the Google Tesseract here) to recognise the text. The recognised text is then added to the input image along with the bounding boxes. 

To improve accuracy and to avoid errors with recognition, the system only recognises text with a font size of atleast 3% of the total image height.

![Flow](https://github.com/muhammedsalihk/Scene-Text-Detection-and-Recognition-App/blob/master/Images/DNN.png)

## Implementation
The webapp has been built using streamlit and can be run using the command 
```
streamlit run myapp.py
```
![App1](https://github.com/muhammedsalihk/Scene-Text-Detection-and-Recognition-App/blob/master/Images/App%201.png)
![App2](https://github.com/muhammedsalihk/Scene-Text-Detection-and-Recognition-App/blob/master/Images/App%202.png)
![App3](https://github.com/muhammedsalihk/Scene-Text-Detection-and-Recognition-App/blob/master/Images/App%203.png)
