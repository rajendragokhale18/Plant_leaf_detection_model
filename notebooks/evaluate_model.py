import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

test_dir = r"C:\Users\91915\Desktop\plantrepo_project\Plant_Disease_Detection\data\test_split"
model_path = r"C:\Users\91915\Desktop\plantrepo_project\Plant_Disease_Detection\models\leaf_model.h5"

img_size = (224, 224)
batch_size = 32

test_datagen = ImageDataGenerator(rescale=1./255)
test_gen = test_datagen.flow_from_directory(
    test_dir, target_size=img_size, batch_size=batch_size, class_mode='categorical'
)

model = tf.keras.models.load_model(model_path)
loss, acc = model.evaluate(test_gen)
print(f"Test accuracy: {acc:.2f}")