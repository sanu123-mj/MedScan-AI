# %%
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

# Use the path where your dataset is stored
data_path = r'C:\Users\lenovo\Desktop\fproject\backend\models\data'

for dirname, _, filenames in os.walk(data_path):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob as gb
from PIL import Image
import os
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPooling2D, Conv2DTranspose, add
from tensorflow.keras import Model
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

# %%
#!pip install kaggle

# %%
import os

# Your local dataset path
dataset_path = r"C:\Users\lenovo\Desktop\fproject\backend\models\data"

# List the first 10 files
for dirname, _, filenames in os.walk(dataset_path):
    for filename in filenames[:10]:  # Just to see the first 10 files
        print(os.path.join(dirname, filename))


# %%
import pandas as pd
import matplotlib.pyplot as plt
import cv2

# Load CSV
df = pd.read_csv('backend/models/data/GroundTruth.csv')


# Pick image name from the correct column
image_name = df['image'].iloc[0]
img_path = f'backend/models/data/images/{image_name}.jpg'


# Load and show the image
print(f"Trying to load: {img_path}")

img = cv2.imread(img_path)

if img is None:
    raise FileNotFoundError(f"Image not found or could not be read: {img_path}")

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.imshow(img)
plt.title("Sample Image")
plt.axis('off')
plt.show()

# %%
#import zipfile

# Path to your ZIP file
#zips

# Path to where you want to extract the contents
#extract_to = r'C:\Users\lenovo\Desktop\fproject\backend\models\data\images'

# Open and extract the ZIP file
#with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
#    zip_ref.extractall(extract_to)

#print("Unzipping done!")


# %%
import os

# Local path to your extracted dataset folder
directory_path = r"C:\Users\lenovo\Desktop\fproject\backend\models\data"

# List all files and folders in the directory
files = os.listdir(directory_path)

# Print the list of files and folders
print("Files in directory:", files)




# %%
import os
import random
import cv2
import matplotlib.pyplot as plt

# Local path to the images directory
images_dir = r"C:\Users\lenovo\Desktop\fproject\backend\models\data\images"

# List all image filenames in the directory
image_files = os.listdir(images_dir)

# Randomly select 25 images
sample_images = random.sample(image_files, 25)


# Plot the 25 images
fig, axes = plt.subplots(5, 5, figsize=(15, 15))

# Loop through the 25 images and display them
for i, ax in enumerate(axes.flat):
    # Read the image
    img_path = os.path.join(images_dir, sample_images[i])
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Display the image
    ax.imshow(img)
    ax.axis('off')
    ax.set_title(sample_images[i])

plt.tight_layout()
plt.show()


# %%
import os

import os

# Update to your actual local paths (after unzipping the dataset)
img_dir = r"C:\Users\lenovo\Desktop\fproject\backend\models\data\images"
mask_dir = r"C:\Users\lenovo\Desktop\fproject\backend\models\data\masks"

# List and sort image and mask files
img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(".jpg")])
mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith(".png")])

# Check if files are loaded correctly
print(f"Found {len(img_files)} images and {len(mask_files)} masks.")


# %%
# Assuming 'img_files' and 'mask_files' are already defined as the list of image and mask filenames.

image_names = [os.path.splitext(f)[0] for f in img_files]  # Extract base names from images
mask_names = [os.path.splitext(f)[0].replace('_segmentation', '') for f in mask_files]  # Extract base names from masks

# Find images without corresponding masks
missing_masks = [f for f in image_names if f not in mask_names]

# Output result
if len(missing_masks) == 0:
    print('No missing masks found.')
else:
    print(f"There are {len(missing_masks)} missing masks found:")
    print(missing_masks)


# %%
model_predictions_dir = '/kaggle/working/model_predictions/'

# %%
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Directories for images and masks (update these to your actual local paths)
img_dir = r"C:\Users\lenovo\Desktop\fproject\backend\models\data\images"
mask_dir = r"C:\Users\lenovo\Desktop\fproject\backend\models\data\masks"

# List of image and mask files
img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(".jpg")])
mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith(".png")])

def display_image_and_mask(n=5, seed=None):
    if seed:
        np.random.seed(seed)

    fig, axs = plt.subplots(2, n, figsize=(20, 6))
    for i in range(n):
        # Randomly select an image index
        idx = np.random.randint(0, len(img_files))
        
        # Construct image and mask file paths
        img_path = os.path.join(img_dir, img_files[idx])
        mask_path = os.path.join(mask_dir, os.path.splitext(img_files[idx])[0] + '_segmentation.png')  # matching mask with image
        
        # Load the image and mask
        img = Image.open(img_path)
        mask = Image.open(mask_path)
        
        # Display image
        axs[0, i].imshow(img)
        axs[0, i].set_title('Image')
        axs[0, i].axis('off')

        # Display mask
        axs[1, i].imshow(mask)
        axs[1, i].set_title('Mask')
        axs[1, i].axis('off')

    # Show the plot
    plt.show()

# Example call to display 5 random images and their corresponding masks
display_image_and_mask(n=5, seed=42)


# %%
# Assuming necessary libraries are already imported

# Directories for images and masks
img_dir = r"C:\Users\lenovo\Desktop\fproject\backend\models\data\images"
mask_dir = r"C:\Users\lenovo\Desktop\fproject\backend\models\data\masks"

# List of image and mask files
img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(".jpg")])
mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith(".png")])

def display_image_with_mask(n=5, seed=None):
    if seed:
        np.random.seed(seed)

    fig, axs = plt.subplots(1, n, figsize=(20, 5))
    for i in range(n):
        # Randomly select an image index
        idx = np.random.randint(0, len(img_files))
        
        # Construct image and mask file paths
        img_path = os.path.join(img_dir, img_files[idx])
        mask_path = os.path.join(mask_dir, os.path.splitext(img_files[idx])[0] + '_segmentation.png')  # matching mask with image
        
        # Load image and mask as numpy arrays
        img_np = np.array(Image.open(img_path))
        mask_np = np.array(Image.open(mask_path))
        
        # Display the image
        axs[i].imshow(img_np)
        
        # Overlay the mask with transparency (alpha = 0.5)
        axs[i].imshow(mask_np, cmap='Reds', alpha=0.5)
        
        # Set title and remove axes
        axs[i].set_title('Image with Mask')
        axs[i].axis('off')

    # Show the plot
    plt.show()

# Example call to display 5 random images with overlaid masks
display_image_with_mask(n=5, seed=42)


# %%
IMG_SIZE = 256
BATCH_SIZE = 32
BUFFER_SIZE = 1000
AUTOTUNE = tf.data.experimental.AUTOTUNE
# img_dir
# mask_dir

# %%
def img_mask_paths(img_dir, mask_dir):
    img_path = sorted(gb.glob(os.path.join(img_dir, '*.jpg')))
    mask_path = sorted(gb.glob(os.path.join(mask_dir, '*.png')))
    return np.array(img_path), np.array(mask_path)

imgs_path, masks_path = img_mask_paths(img_dir, mask_dir)

# %%
X_train, X_temp, y_train, y_temp = train_test_split(imgs_path, masks_path, train_size=0.90, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, train_size=0.5, random_state=42)

# %%
print(f"{'Training:':<15}{len(X_train)}")
print(f"{'Validation:':<15}{len(X_val)}")
print(f"{'Testing:':<15}{len(X_test)}")

# %%
import tensorflow as tf

# Disable all GPUs
tf.config.set_visible_devices([], 'GPU')

# Optionally, check to confirm TensorFlow is using only the CPU
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


# %%
def map_fn(img_path, mask_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    img = tf.cast(img, tf.float32) / 255.0

    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_jpeg(mask, channels=1)
    mask = tf.image.resize(mask, (IMG_SIZE, IMG_SIZE))
    mask = tf.cast(mask, tf.float32) / 255.0

    return img, mask

# %%
# train_set
train_set = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_set = train_set.map(map_fn, num_parallel_calls=AUTOTUNE)
train_set = train_set.shuffle(buffer_size=BUFFER_SIZE).batch(batch_size=BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)

# val_set
val_set = tf.data.Dataset.from_tensor_slices((X_val, y_val))
val_set = val_set.map(map_fn, num_parallel_calls=AUTOTUNE)
val_set = val_set.shuffle(buffer_size=BUFFER_SIZE).batch(batch_size=BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)

# test_set
test_set = tf.data.Dataset.from_tensor_slices((X_test, y_test))
test_set = test_set.map(map_fn, num_parallel_calls=AUTOTUNE)
test_set = test_set.shuffle(buffer_size=BUFFER_SIZE).batch(batch_size=BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)

# %%
# U-Net Model
def UNET():
    inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 3))

    x = Conv2D(32, 3, strides=1, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    skip_connections = []

    # Encoder
    for filters in [64, 128, 256, 512]:
        x = Conv2D(filters, 3, strides=1, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Conv2D(filters, 3, strides=1, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        skip_connections.append(x)
        x = MaxPooling2D(2, strides=2, padding='same')(x)

    # Bottleneck
    x = Conv2D(1024, 3, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(1024, 3, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Decoder
    for filters in [512, 256, 128, 64]:
        x = Conv2DTranspose(filters, 2, strides=2, padding='same')(x)
        skip_connection = skip_connections.pop()
        x = add([x, skip_connection])

        x = Conv2D(filters, 3, strides=1, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Conv2D(filters, 3, strides=1, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

    outputs = Conv2D(1, 1, strides=1, activation='sigmoid')(x)

    model = Model(inputs=[inputs], outputs=[outputs])
    return model

# %%
model = UNET()
model.summary()
Model: "functional"

# %%
def dice_coefficient(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return 1 - dice_coefficient(y_true, y_pred)

# %%
model.compile(optimizer=Adam(0.0002),
              loss=BinaryCrossentropy(),
              metrics=['accuracy', Precision(name='precision'), Recall(name='recall'), dice_coefficient])

# %%
checkpoint_cb = ModelCheckpoint('best_weights.weights.h5',
                                monitor='val_loss',
                                verbose=1,
                                save_best_only=True,
                                save_weights_only=True)

reduce_lr_cb = ReduceLROnPlateau(monitor='val_loss',
                                 factor=0.1,
                                 patience=5,
                                 verbose=1)

# %%
history = model.fit(train_set, epochs=1, validation_data=val_set, callbacks=[checkpoint_cb, reduce_lr_cb])

# %%
# Permanent model saving additions (only these 3 new lines at the very end)
MODEL_SAVE_PATH = r"C:\Users\lenovo\Desktop\fproject\backend\models\permanent_skin_model.h5"
model.save(MODEL_SAVE_PATH)
print(f"Model permanently saved at: {MODEL_SAVE_PATH}")