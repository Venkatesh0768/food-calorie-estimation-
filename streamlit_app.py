import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Load the trained model
MODEL_PATH = r'D:\Food cal 2\models\food_classifier.h5'
model = tf.keras.models.load_model(MODEL_PATH)

# Load the calorie mapping
calorie_df = pd.read_csv(r'D:\Food cal 2\data\calorie_mapping.csv')
calorie_dict = calorie_df.set_index('Food Item').to_dict()['Calories']

# List of food classes (should match the model's output classes)
food_classes = [
    'apple_pie', 'baby_back_ribs', 'baklava', 'beef_carpaccio', 'beef_tartare', 'beet_salad',
    'beignets', 'bibimbap', 'bread_pudding', 'breakfast_burrito', 'bruschetta', 'caesar_salad',
    'cannoli', 'caprese_salad', 'carrot_cake', 'ceviche', 'cheesecake', 'chicken_curry',
    'chicken_wings', 'chili', 'chocolate_cake', 'churros', 'clams', 'coq_au_vin',
    'croque_madame', 'cup_cakes', 'deviled_eggs', 'donuts', 'dumplings', 'edamame',
    'egg_salad', 'eggs_benedict', 'enchiladas', 'falafel', 'fettuccine_alfredo', 'fish_tacos',
    'foie_gras', 'french_fries', 'french_onion_soup', 'frozen_yogurt', 'garlic_bread',
    'goulash', 'grilled_cheese', 'guacamole', 'hamburger', 'hot_and_sour_soup', 'hot_dog',
    'hummus', 'ice_cream', 'macaroni_and_cheese', 'macarons', 'mango_smoothie', 'meatballs',
    'mussels', 'nachos', 'pad_thai', 'paella', 'pancakes', 'pasta',
    'pasta_bake', 'pasta_salad', 'pepperoni_pizza', 'pho', 'pizza', 'pork_chop',
    'pork_ribs', 'pot_pie', 'poutine', 'ramen', 'red_velvet_cake', 'risotto',
    'samosa', 'sandwich', 'sauerbraten', 'seafood_pasta', 'shrimp_and_grits', 'sushi',
    'tacos', 'takoyaki', 'tiramisu', 'tofu', 'tortilla', 'turkey',
    'veggie_burger', 'vegetarian_chili', 'waffles', 'zucchini_bread', 'zucchini_soup'
]

def load_image(img):
    # Preprocess the image
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img /= 255.0  # Normalize image
    return img

def estimate_calories(img):
    # Preprocess and predict
    img = load_image(img)
    preds = model.predict(img)
    
    # Get the class index with the highest probability
    class_index = np.argmax(preds, axis=1)[0]
    
    # Map the class index to food name
    food_item = map_index_to_food_name(class_index)
    calories = calorie_dict.get(food_item, "Calorie info not available")
    
    return food_item, calories

def map_index_to_food_name(index):
    if index < len(food_classes):
        return food_classes[index]
    else:
        return "Unknown food item"

# Streamlit App
st.title("Food Calorie Estimator")

# File uploader allows users to upload images
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load the image
    img = image.load_img(uploaded_file, target_size=(224, 224))
    
    # Display the image
    st.image(img, caption='Uploaded Image', use_column_width=True)
    
    # Estimate calories
    food_item, calories = estimate_calories(img)
    
    # Display the results
    st.write(f"**Predicted Food Item:** {food_item}")
    st.write(f"**Estimated Calories:** {calories} kcal")

