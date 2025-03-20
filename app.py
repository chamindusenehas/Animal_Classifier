from flask import Flask,render_template,jsonify,request
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
import json
from werkzeug.utils import secure_filename
import os
from flask_cors import CORS


app = Flask(__name__)
CORS(app)


model_path = 'static/animal_classifier_model.h5'
json_path = 'static/class_names.json'
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)



#Load Class names
with open(json_path, 'r') as f:
    class_indices = json.load(f)
class_names = list(class_indices.keys())
names = ''
for name in class_names:
    names += f'{name}\n'


# Load Model
model = load_model(model_path)
print("Model loaded successfully.")


# Get the Prediction
def prediction(image_path):
    img = image.load_img(image_path, target_size=(224, 224)) 
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array) 

    predictions = model.predict({'input_layer_2': img_array})
    predicted_class_idx = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_names[predicted_class_idx]
    confidence = round(predictions[0][predicted_class_idx] * 100,2)

    return predicted_class_name, confidence






# Home Route
@app.route('/')
def home():
    return render_template('index.html')


# Prediction API
@app.route('/predict',methods=['POST'])
def animal():
    file = request.files['image']
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    animal_name,confidence = prediction(filepath)


    return jsonify({
        'success':True,
        'animal': animal_name,
        'confidence': confidence
        })





if __name__ == '__main__' :
    app.run(debug=True)
    