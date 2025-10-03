from PIL import Image
import matplotlib.pyplot as plt
import os

train_dir = r"C:\Users\91915\Desktop\plantrepo_project\Plant_Disease_Detection\data\Train"

print("Classes in train_dir:", os.listdir(train_dir))

found_image = False

for class_folder in os.listdir(train_dir):
    class_path = os.path.join(train_dir, class_folder)
    if not os.path.isdir(class_path):
        continue
    # Recursively search for images in all subfolders
    for root, dirs, files in os.walk(class_path):
        print(f"Checking {root}...")
        images = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        print(f"  Found {len(images)} images: {images}")
        if images:
            img_path = os.path.join(root, images[0])
            print("Testing with:", img_path)
            img = Image.open(img_path).convert('RGB')
            print(f"Image size: {img.size}, mode: {img.mode}")
            plt.imshow(img)
            plt.axis('off')
            plt.show()
            found_image = True
            break
    if found_image:
        break

if not found_image:
    print("No images found in any class folder or subfolder.")