import tensorflow as tf
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt

model_path = r"C:\Users\91915\Desktop\plantrepo_project\Plant_Disease_Detection\models\leaf_model.h5"
img_path = r"C:\Users\91915\Desktop\plantrepo_project\Plant_Disease_Detection\data\Test\Test\Powdery\9fa9b13467c0961d.jpg"  # Change this to your image

img_size = (224, 224)

model = tf.keras.models.load_model(model_path)

# Get class labels from the test generator
test_dir = r"C:\Users\91915\Desktop\plantrepo_project\Plant_Disease_Detection\data\test_split"
from tensorflow.keras.preprocessing.image import ImageDataGenerator
test_datagen = ImageDataGenerator(rescale=1./255)
test_gen = test_datagen.flow_from_directory(
    test_dir, target_size=img_size, batch_size=1, class_mode='categorical', shuffle=False
)
class_labels = list(test_gen.class_indices.keys())

img = Image.open(img_path).convert('RGB').resize(img_size)
img_array = np.array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

pred = model.predict(img_array)
pred_class = np.argmax(pred)
pred_label = class_labels[pred_class]
print("Predicted:", pred_label)

# Split label for plant type and health status if using '___' format
if '___' in pred_label:
    plant, status = pred_label.split('___')
    print(f"Plant: {plant}, Health Status: {status}")
else:
    print(f"Health Status: {pred_label}")

# Show the image with prediction
plt.imshow(np.array(img))
plt.title(f"Predicted: {pred_label}")
plt.axis('off')
plt.show()