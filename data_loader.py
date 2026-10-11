import os
import numpy as np
import matplotlib.pyplot as plt
import config


def load_ford_txt(filepath: str):

    raw = np.loadtxt(filepath)             
    y = raw[:, 0].astype(np.int32)       
    X = raw[:, 1:].astype(np.float32)    
    
    return X, y

def load_dataset(data_dir, dataset_name="FordB"):

    train_path = os.path.join(data_dir, f"{dataset_name}_TRAIN.txt")
    test_path  = os.path.join(data_dir, f"{dataset_name}_TEST.txt")
    
    print(f"Loading {dataset_name} data...")
    X_train, y_train = load_ford_txt(train_path)
    X_test, y_test   = load_ford_txt(test_path)
    
    print("Data Loaded Successfully")
    print(f"Train shapes: X = {X_train.shape}, y = {y_train.shape}")
    print(f"Test shapes:  X = {X_test.shape}, y = {y_test.shape}\n")
    
    return X_train, y_train, X_test, y_test



def check_and_plot_balance(y_train, y_test, dataset_name="FordB"):

    splits = {"TRAIN": y_train, "TEST": y_test}
    print(f"\nClass Balance Check ({dataset_name}) ")
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(f"Class Distribution - {dataset_name}", fontweight="bold")
    
    for ax, (name, y) in zip(axes, splits.items()):
        n_healthy = (y == 1).sum()
        n_faulty  = (y == -1).sum()
        total     = len(y)
        
        print(f"{name} -> Healthy: {n_healthy} ({n_healthy/total*100:.1f}%) | Faulty: {n_faulty} ({n_faulty/total*100:.1f}%)")
        
        
        ax.bar(["Healthy (+1)", "Faulty (-1)"], [n_healthy, n_faulty], 
               color=[config.COLORS["healthy"], config.COLORS["faulty"]])
        ax.set_title(name)
        ax.set_ylabel("Count")
        
    save_path = f"{config.OUTPUT_DIR}/{dataset_name}_balance.png"
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"saved in: {save_path}\n")

"""
if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_dataset(config.DATA_DIR, dataset_name="FordB")
    
    check_and_plot_balance(y_train, y_test, dataset_name="FordA")
"""