import os



def setup_required(verbose:bool = False):
    
    required_folders = [
        "./data",
        "./data/reports",
        "./data/reports/runs",
        "./data/samples",
        "./data/saved_models",
        "./data/model_registry"
    ]
    
    for folder_path in required_folders:
        if not os.path.isdir(folder_path):
            print(f"{folder_path} not found, creating folder..")
            os.mkdir(folder_path)
        else:
            if verbose:
                print(f"\U0001F44D Folder {folder_path} confirmed")


if __name__ == '__main__':
    setup_required()