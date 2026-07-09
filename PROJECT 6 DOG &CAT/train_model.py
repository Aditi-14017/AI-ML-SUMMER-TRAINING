import os
import cv2
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

# Dataset path
dataset_path = r"/content/sample_data/ImageDataset"

# Create dummy directories and files if they don't exist
for class_name in ["Cat", "Dog"]:
    class_dir = os.path.join(dataset_path, class_name)
    os.makedirs(class_dir, exist_ok=True)
    # Create dummy image files
    for i in range(5):
        dummy_image_path = os.path.join(class_dir, f"image_{i}.png")
        # Create a blank image using numpy and save it with opencv
        dummy_image = np.zeros((64, 64, 3), dtype=np.uint8)
        cv2.imwrite(dummy_image_path, dummy_image)

images = []
labels = []

classes = ["Cat", "Dog"]

IMG_SIZE = 64

for label, folder in enumerate(classes):

    folder_path = os.path.join(dataset_path, folder)

    for file in os.listdir(folder_path):

        img_path = os.path.join(folder_path, file)

        img = cv2.imread(img_path)

        if img is None:
            continue

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        img = img.flatten()

        images.append(img)

        labels.append(label)


X = np.array(images)
y = np.array(labels)

print("Training Images", len(X))

model = LogisticRegression(max_iter=1000)

model.fit(X, y)

joblib.dump(model, "cat_dog_model.pkl")

print("Model Saved Successfully!")

