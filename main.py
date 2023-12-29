import streamlit as st
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
import numpy as np

# Load the pre-trained VGG16 model
model = VGG16(weights='imagenet')

def predict_label(img_path):
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Get model predictions
    predictions = model.predict(img_array)
    decoded_predictions = decode_predictions(predictions)

    # Display the top prediction
    label = decoded_predictions[0][0][1]
    confidence = decoded_predictions[0][0][2]
    return label, confidence

def main():
    st.title("VGGNet Image Classification App")

    # Upload image through Streamlit
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

        # Perform prediction
        label, confidence = predict_label(uploaded_file)

        # Display the result
        st.write(f"Prediction: {label}")
        st.write(f"Confidence: {confidence:.2%}")

if __name__ == "__main__":
    main()
