import os

dataset_path = r"C:\Users\sandh\OneDrive\Desktop\Plant_disease"

for folder in os.listdir(dataset_path):
    folder_path = os.path.join(dataset_path, folder)

    if os.path.isdir(folder_path):
        print(folder)

        count = len(os.listdir(folder_path))

        print("Images:", count)
        print()