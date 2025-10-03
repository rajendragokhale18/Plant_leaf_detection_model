import cv2
import tensorflow as tf
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Paths
model_path = r"C:\Users\91915\Desktop\plantrepo_project\Plant_Disease_Detection\models\leaf_model.h5"
test_dir = r"C:\Users\91915\Desktop\plantrepo_project\Plant_Disease_Detection\data\test_split"
img_size = (224, 224)

# Load model
model = tf.keras.models.load_model(model_path)

# Get class labels from the test generator
test_datagen = ImageDataGenerator(rescale=1. / 255)
test_gen = test_datagen.flow_from_directory(
    test_dir, target_size=img_size, batch_size=1, class_mode='categorical', shuffle=False
)
class_labels = list(test_gen.class_indices.keys())


# Function to detect leaf using HSV + contours
def detect_leaf(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define range for green color in HSV
    lower_green = np.array([25, 40, 40])
    upper_green = np.array([85, 255, 255])

    # Create mask
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Morphological operations to clean noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Find contours on mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        return x, y, w, h, mask
    return None, None, None, None, mask


# Open camera
cap = cv2.VideoCapture(1)  # Change index if needed
if not cap.isOpened():
    print("Cannot open camera")
    exit()

print("Press SPACE to capture and predict, ESC to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame. Exiting ...")
        break

    # Detect leaf and draw bounding box
    x, y, w, h, mask = detect_leaf(frame)
    if x is not None:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imshow("Mask", mask)  # Debug window for green detection

    cv2.imshow('Camera', frame)
    key = cv2.waitKey(1)

    if key % 256 == 27:  # ESC pressed
        print("Escape hit, closing...")
        break
    elif key % 256 == 32 and x is not None:  # SPACE pressed, and leaf detected
        # Crop leaf region
        leaf_crop = frame[y:y + h, x:x + w]
        img = cv2.cvtColor(leaf_crop, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img).resize(img_size)
        img_array = np.array(img_pil) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict disease
        pred = model.predict(img_array)
        pred_class = np.argmax(pred)
        pred_label = class_labels[pred_class]
        print("Predicted Disease:", pred_label)

        # Show prediction
        plt.imshow(img_pil)
        plt.title(f"Predicted Disease: {pred_label}")
        plt.axis('off')
        plt.show()

        # TODO: Species classification model (future work)
        # pred_species = species_model.predict(img_array)
        # species_label = species_classes[np.argmax(pred_species)]
        # print("Detected Species:", species_label)

cap.release()
cv2.destroyAllWindows()
