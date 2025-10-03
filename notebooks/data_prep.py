import os
from sklearn.model_selection import train_test_split
import shutil

def split_data(source_dir, train_dir, val_dir, test_dir, val_size=0.15, test_size=0.15):
    print(f"Source directory (absolute): {os.path.abspath(source_dir)}")
    print("Folders/files in source_dir:", os.listdir(source_dir))
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    for class_name in os.listdir(source_dir):
        class_path = os.path.join(source_dir, class_name)
        print(f"Found folder: {class_name} (isdir: {os.path.isdir(class_path)})")
        if not os.path.isdir(class_path):
            continue
        print("  Files in this folder:", os.listdir(class_path))
        images = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        print(f"{class_name}: {len(images)} images found")
        if not images:
            continue
        train_imgs, temp_imgs = train_test_split(images, test_size=val_size+test_size, random_state=42)
        val_imgs, test_imgs = train_test_split(temp_imgs, test_size=test_size/(val_size+test_size), random_state=42)
        print(f"  Train: {len(train_imgs)}, Val: {len(val_imgs)}, Test: {len(test_imgs)}")
        for img_list, target_dir in zip([train_imgs, val_imgs, test_imgs], [train_dir, val_dir, test_dir]):
            class_target = os.path.join(target_dir, class_name)
            os.makedirs(class_target, exist_ok=True)
            for img in img_list:
                shutil.copy(os.path.join(class_path, img), os.path.join(class_target, img))

if __name__ == "__main__":
    split_data(
        source_dir=r"C:\Users\91915\Desktop\plantrepo_project\Plant_Disease_Detection\data\Train\Train",
        train_dir=r"C:\Users\91915\Desktop\plantrepo_project\Plant_Disease_Detection\data\train_split",
        val_dir=r"C:\Users\91915\Desktop\plantrepo_project\Plant_Disease_Detection\data\val_split",
        test_dir=r"C:\Users\91915\Desktop\plantrepo_project\Plant_Disease_Detection\data\test_split"
    )