import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load the model
model = load_model('model.h5')

# List of all labels
all_labels = ['Speed limit (20km/h)','Speed limit (30km/h)','Speed limit (50km/h)','Speed limit (60km/h)',
              'Speed limit (70km/h)','Speed limit (80km/h)','End of speed limit (80km/h)','Speed limit (100km/h)',
              'Speed limit (120km/h)','No passing','No passing for vehicles over 3.5 metric tons',
              'Right-of-way at the next intersection','Priority road','Yield','Stop','No vehicles',
              'Vehicles over 3.5 metric tons prohibited','No entry','General caution','Dangerous curve to the left',
              'Dangerous curve to the right','Double curve','Bumpy road','Slippery road','Road narrows on the right',
              'Road work','Traffic signals','Pedestrians','Children crossing','Bicycles crossing','Beware of ice/snow',
              'Wild animals crossing','End of all speed and passing limits','Turn right ahead','Turn left ahead',
              'Ahead only','Go straight or right','Go straight or left','Keep right','Keep left','Roundabout mandatory',
              'End of no passing','End of no passing by vehicles over 3.5 metric tons']

# Streamlit app
st.title('Traffic Sign Classification by yuva')

st.write("Upload an image of a traffic sign to get its classification:")

# File uploader widget
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Load and preprocess the image
    img = Image.open(uploaded_file)
    img = img.resize((50, 50))  # Resize image to match model's expected input size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize the image

    # Display the preprocessed image for debugging
    st.image(img_array[0], caption='Preprocessed Image', use_column_width=True)
    
    # Make prediction
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])

    # Display the image and prediction
    st.image(img, caption='Uploaded Image', use_column_width=True)
    st.write(f'Predicted class index: {predicted_class}')
    st.write(f'Predicted label: {all_labels[predicted_class]}')

    # Debug print statements
    st.write("Prediction probabilities:", predictions)
    st.write("Predicted class index:", predicted_class)
