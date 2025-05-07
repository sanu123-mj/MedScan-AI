from tensorflow.keras.models import load_model
import os

# Step 1: Ensure the directory to save the model exists
os.makedirs('path_to_save_model/', exist_ok=True)  # Replace with the actual path where you want to save the model

# Step 2: Load the trained model
model = load_model('best_weights.weights.h5')  # Use your actual model file path here

# Step 3: Save the model to the new location
model.save('path_to_save_model/skin_model_saved.h5')  # Path where you want to save the model

print("Model saved successfully!")
