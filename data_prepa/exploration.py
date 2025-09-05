import os
import matplotlib.pyplot as plt
import numpy as np

possible_paths = [
    "data\CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone\CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone",
    "data\CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone",
    "data\CT-KIDNEY-DATASET"
]

dataset_path = None
for path in possible_paths:
    if os.path.exists(path):
        if any(os.path.isdir(os.path.join(path, class_name)) for class_name in ["Cyst", "Normal", "Stone", "Tumor"]):
            dataset_path = path
            break

if dataset_path is None:
    print("ERREUR: Impossible de trouver le dataset. Voici la structure actuelle du dossier:")
    for root, dirs, files in os.walk("data"):
        print(f"Dossier: {root}")
        for d in dirs:
            print(f"  Sous-dossier: {d}")
        if len(files) < 10:
            for f in files:
                print(f"  Fichier: {f}")
        else:
            print(f"  {len(files)} fichiers trouvés dans ce dossier")
    print("\nVeuillez vérifier le chemin correct et modifier la variable dataset_path.")
else:
    print(f"Dataset trouvé au chemin: {dataset_path}")
    
    class_counts = {}
    classes_found = False
    for class_name in ["Cyst", "Normal", "Stone", "Tumor"]:
        class_path = os.path.join(dataset_path, class_name)
        if os.path.isdir(class_path):
            classes_found = True
            image_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp', '.tif', '.tiff'))]
            class_counts[class_name] = len(image_files)
            print(f"Trouvé {len(image_files)} images dans la classe {class_name}")
    
    if not classes_found:
        print("ERREUR: Aucune des classes attendues n'a été trouvée dans le dossier.")
    elif not class_counts:
        print("ERREUR: Aucune image n'a été trouvée dans les dossiers de classes.")
    else:
        total_images = sum(class_counts.values())
        print(f"\nTotal d'images dans le dataset: {total_images}")
        print("\nDistribution des classes:")
        for class_name, count in class_counts.items():
            percentage = (count / total_images) * 100
            print(f"{class_name}: {count} images ({percentage:.2f}%)")
        
        if class_counts:
            min_count = min(class_counts.values())
            max_count = max(class_counts.values())
            ratio = max_count / min_count if min_count > 0 else float('inf')
            
            print(f"\nRatio entre la classe la plus nombreuse et la moins nombreuse: {ratio:.2f}")
            
            if ratio > 1.5:
                print("DÉSÉQUILIBRE DÉTECTÉ: Le ratio est supérieur à 1.5, ce qui peut affecter les performances du modèle.")
                print("Techniques recommandées pour gérer le déséquilibre:")
                print("1. Augmentation de données plus intensive pour les classes minoritaires")
                print("2. Sous-échantillonnage des classes majoritaires")
                print("3. Utilisation de poids de classe dans la fonction de perte")
                print("4. Techniques de sur-échantillonnage comme SMOTE")
            
            plt.figure(figsize=(10, 6))
            classes = list(class_counts.keys())
            counts = list(class_counts.values())
            bars = plt.bar(classes, counts, color=['skyblue', 'lightgreen', 'salmon', 'purple'])
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 5,
                         f'{height} ({height/total_images*100:.1f}%)',
                         ha='center', va='bottom')
            plt.title('Distribution des classes dans le dataset de scanners rénaux')
            plt.xlabel('Classes')
            plt.ylabel('Nombre d\'images')
            plt.ylim(0, max(counts) * 1.2)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.savefig('class_distribution.png')
            plt.close()
            print("\nGraphique de distribution sauvegardé sous 'class_distribution.png'")
            
            with open("class_distribution_analysis.txt", "w") as f:
                f.write(f"ANALYSE DE LA DISTRIBUTION DES CLASSES\n")
                f.write(f"====================================\n\n")
                f.write(f"Total d'images dans le dataset: {total_images}\n\n")
                f.write("Distribution des classes:\n")
                for class_name, count in class_counts.items():
                    percentage = (count / total_images) * 100
                    f.write(f"{class_name}: {count} images ({percentage:.2f}%)\n")
                f.write(f"\nRatio entre la classe la plus nombreuse et la moins nombreuse: {ratio:.2f}\n")
                if ratio > 1.5:
                    f.write("\nDÉSÉQUILIBRE DÉTECTÉ: Le ratio est supérieur à 1.5\n")
                    f.write("Recommandations pour gérer le déséquilibre:\n")
                    f.write("1. Augmentation de données plus intensive pour les classes minoritaires\n")
                    f.write("2. Sous-échantillonnage des classes majoritaires\n")
                    f.write("3. Utilisation de poids de classe dans la fonction de perte\n")
                    f.write("4. Techniques de sur-échantillonnage comme SMOTE\n")
            print("Analyse complète sauvegardée dans 'class_distribution_analysis.txt'")
