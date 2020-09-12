import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from local_utils import detect_lp
from os.path import splitext, basename
from keras.models import model_from_json
import glob

from keras.preprocessing.image import load_img, img_to_array
from keras.layers import Input
from keras.layers import *
from keras.models import Model
from sklearn.preprocessing import LabelEncoder

UPLOAD_FOLDER = 'static/'
WPOD_NET_PATH = 'model/wpod-net.json'

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

class Vpd:
    def __init__(self):
        self.model = None
        self.labels = None
        self.wpod_net = None
   
    def load_wpod_net(self):
        if self.wpod_net is None:
            try:
                path = splitext(WPOD_NET_PATH)[0]
                with open('%s.json' % path, 'r') as json_file:
                    model_json = json_file.read()
                self.wpod_net = model_from_json(model_json, custom_objects={})
                self.wpod_net.load_weights('%s.h5' % path)
            except Exception as e:
                print(e)
        return self.wpod_net

    def preprocess_image(self, image_path,resize=False):
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img / 255
        if resize:
            img = cv2.resize(img, (224,224))
        return img


    def get_plate(self, image_path, Dmax=608, Dmin=270):
        vehicle = self.preprocess_image(image_path)
        ratio = float(max(vehicle.shape[:2])) / min(vehicle.shape[:2])
        side = int(ratio * Dmin)
        bound_dim = min(side, Dmax)
        _ , LpImg, _, cor = detect_lp(self.load_wpod_net(), vehicle, bound_dim, lp_threshold=0.5)

        return LpImg, cor

    def extractPlate(self, multiple_plates_image):
        LpImg,cor = self.get_plate(multiple_plates_image)
        x1 = int(min(cor[0][1]))
        x2 = int(max(cor[0][1]))
        y1 = int(min(cor[0][0]))
        y2 = int(max(cor[0][0]))
        im = self.preprocess_image(multiple_plates_image)
        crop_image = im[x1:x2, y1:y2]
        height, width, channels = crop_image.shape
        
        #Resizing the image
        crop_image = cv2.resize(crop_image,(width, height))
        aratio = width/height;
        LpImg[0] = crop_image
        if aratio <=3:
            LpImg[0] = im[x1:x1 + int(height/2)+2, y1:y2] # Top
            if len(LpImg) == 1:
                LpImg.append(None)
            LpImg[1] = im[x1 + int(height/2):x2, y1:y2] # Bottom
        
        return LpImg

    # Create sort_contours() function to grab the contour of each digit from left to right
    def sort_contours(self, cnts,reverse = False):
        i = 0
        boundingBoxes = [cv2.boundingRect(c) for c in cnts]
        (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes), key=lambda b: b[1][i], reverse=reverse))
        return cnts

    def photoShopImage(self, lpimg):
        # Initialize a list which will be used to append charater image
        crop_characters = []
        # Scales, calculates absolute values, and converts the result to 8-bit.
        plate_image = cv2.convertScaleAbs(lpimg, alpha=(255.0))
        
        # convert to grayscale and blur the image
        gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray,(7,7),0)
        
        # Applied inversed thresh_binary 
        binary = cv2.threshold(blur, 180, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        ## Applied dilation 
        kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thre_mor = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel3)
        cont, _  = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
        
        # creat a copy version "test_roi" of plat_image to draw bounding box
        test_roi = plate_image.copy()
        # define standard width and height of character
        digit_w, digit_h = 30, 60
        for c in self.sort_contours(cont):
            (x, y, w, h) = cv2.boundingRect(c)
            ratio = h/w
            if 1<=ratio<=3.5: # Only select contour with defined ratio
                if h/plate_image.shape[0]>=0.5: # Select contour which has the height larger than 50% of the plate
                    # Draw bounding box arroung digit number
                    cv2.rectangle(test_roi, (x, y), (x + w, y + h), (0, 255,0), 2)
                    # Sperate number and gibe prediction
                    curr_num = thre_mor[y:y+h,x:x+w]
                    curr_num = cv2.resize(curr_num, dsize=(digit_w, digit_h))
                    _, curr_num = cv2.threshold(curr_num, 220, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    crop_characters.append(curr_num)

        return crop_characters

    def loadModel(self):
        # Load model architecture, weight and labels
        json_file = open('model/mobileNets.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        model.load_weights("model/weight.h5")
        labels = LabelEncoder()
        labels.classes_ = np.load('model/classes.npy')
        model_set=True
        return model,labels

    # pre-processing input images and pedict with model
    def predict_from_model(self, image,model,labels):
        image = cv2.resize(image,(80,80))
        image = np.stack((image,)*3, axis=-1)
        prediction = labels.inverse_transform([np.argmax(model.predict(image[np.newaxis,:]))])
        return prediction

    def read_character(self, img):
        final_string = ''
        if self.model is None or self.labels is None:
            self.model,self.labels = self.loadModel()

        crop_characters = self.photoShopImage(img)
        for i,character in enumerate(crop_characters):
            title = np.array2string(self.predict_from_model(character,self.model,self.labels))
            final_string+=title.strip("'[]")
        
        return final_string

detector = Vpd()

from flask import Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# initialize our Flask application
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/")
def index():
    return "<h1>Welcome to our server !!</h1>"

@app.route("/image", methods=["GET", "POST"])
def uploadImage():
    plate=""
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            #After uploading the file we process it
            LpImg = detector.extractPlate('static/' + filename)
            for img in LpImg:
                plate += detector.read_character(img)

            src = url_for('static', filename=filename)
            if plate!="":
                plate = "<h3>DETECTED: "+plate+"</h3><img width='40%' height='40%' src='"+src+"'/>"
            #return redirect(url_for('uploadImage',filename=filename))

    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    <hr />
    ''' + plate

if __name__ == "__main__":
    app.run(threaded=True, debug=True)
