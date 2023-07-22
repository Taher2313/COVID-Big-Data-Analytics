import os

def delete_files_in_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
            print(f"Deleted file: {file_path}")

# Specify the folders you want to delete files from
folders = ["dataset/02-prepared", "dataset/03-splited", "dataset/04-preprocessed", "dataset/05-naive-bayes"]
# Iterate over the folders and delete files in each one
for folder in folders:
    delete_files_in_folder(folder)