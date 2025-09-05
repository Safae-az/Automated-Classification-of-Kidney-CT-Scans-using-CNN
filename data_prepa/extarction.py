import os
import zipfile

# Extraction du dataset
zip_path = "archive.zip"
extract_path = "data"
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)
print("Extraction terminée !")

# Count des images dans chaque classe
dataset_path = os.path.join(extract_path,"-Normal-Cyst-Tumor-Stone")

class_names = ["Cyst", "Normal", "stone", "Tumor"]

with open("image_count.txt", "w") as f:
    for class_name in class_names:
        class_path = os.path.join(dataset_path, class_name)
        if os.path.isdir(class_path):
            num_images = len([img for img in os.listdir(class_path) if img.lower().endswith(('.jpg', '.png', '.jpeg'))])
            f.write(f"{class_name} : {num_images} images\n")
            print(f"{class_name} : {num_images} images")  # Ajouté pour feedback immédiat