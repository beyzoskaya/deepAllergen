import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import hashlib
import os

TRAIN_PATH = "preprocessed_data/train_dataset"
VAL_PATH   = "preprocessed_data/val_dataset"
TEST_PATH  = "preprocessed_data/test_dataset"

OUTPUT_DIR = "dataset_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_SIMILARITY_SAMPLES = 3000   
BATCH_SIZE = 32

def load_dataset(path):
    ds = tf.data.Dataset.load(path)
    ds = ds.batch(BATCH_SIZE)
    return ds


def dataset_to_numpy(ds):
    X_all = []
    y_all = []

    for x, y in tqdm(ds):
        X_all.append(x.numpy())
        y_all.append(y.numpy())

    return np.concatenate(X_all), np.concatenate(y_all)


def get_sequence_hash(seq):
    """
    Used for leakage detection.
    """
    return hashlib.md5(seq.tobytes()).hexdigest()


def sample_data(X, y, max_samples):
    if len(X) > max_samples:
        idx = np.random.choice(len(X), max_samples, replace=False)
        return X[idx], y[idx]
    return X, y

print("Loading datasets...")

train_ds = load_dataset(TRAIN_PATH)
val_ds   = load_dataset(VAL_PATH)
test_ds  = load_dataset(TEST_PATH)

print("Converting to numpy...")

X_train, y_train = dataset_to_numpy(train_ds)
X_val, y_val     = dataset_to_numpy(val_ds)
X_test, y_test   = dataset_to_numpy(test_ds)

print("Train:", X_train.shape)
print("Val:", X_val.shape)
print("Test:", X_test.shape)

def plot_label_distribution():

    data = {
        "Train": y_train,
        "Validation": y_val,
        "Test": y_test
    }

    rows = []
    for name, labels in data.items():
        rows.append({
            "Dataset": name,
            "Allergen": np.sum(labels == 1),
            "Non-Allergen": np.sum(labels == 0)
        })

    df = pd.DataFrame(rows)
    print("\nLabel Distribution")
    print(df)

    df.set_index("Dataset").plot(kind="bar")
    plt.title("Label Distribution")
    plt.ylabel("Count")
    plt.savefig(f"{OUTPUT_DIR}/label_distribution.png")
    plt.close()


plot_label_distribution()

def plot_sequence_lengths():

    train_len = [x.shape[0] for x in X_train]
    val_len   = [x.shape[0] for x in X_val]
    test_len  = [x.shape[0] for x in X_test]

    plt.hist(train_len, bins=50, alpha=0.5, label="Train")
    plt.hist(val_len, bins=50, alpha=0.5, label="Val")
    plt.hist(test_len, bins=50, alpha=0.5, label="Test")

    plt.title("Sequence Length Distribution")
    plt.legend()
    plt.savefig(f"{OUTPUT_DIR}/length_distribution.png")
    plt.close()


plot_sequence_lengths()

def leakage_check():

    print("\n🔍 Checking duplicates...")

    def hash_set(X):
        return set(get_sequence_hash(seq) for seq in X)

    train_hash = hash_set(X_train)
    val_hash   = hash_set(X_val)
    test_hash  = hash_set(X_test)

    train_val = len(train_hash & val_hash)
    train_test = len(train_hash & test_hash)
    val_test = len(val_hash & test_hash)

    print("\nPossible Leakage")
    print("Train ↔ Val overlap:", train_val)
    print("Train ↔ Test overlap:", train_test)
    print("Val ↔ Test overlap:", val_test)


leakage_check()

def flatten_embedding(X):
    return X.reshape(X.shape[0], -1)


def similarity_analysis():

    print("\nSimilarity Analysis")

    Xt, yt = sample_data(X_train, y_train, MAX_SIMILARITY_SAMPLES)
    Xv, yv = sample_data(X_val, y_val, MAX_SIMILARITY_SAMPLES)
    Xte, yte = sample_data(X_test, y_test, MAX_SIMILARITY_SAMPLES)

    Xt_f = flatten_embedding(Xt)
    Xv_f = flatten_embedding(Xv)
    Xte_f = flatten_embedding(Xte)

    sim_train_val = cosine_similarity(Xt_f[:500], Xv_f[:500]).mean()
    sim_train_test = cosine_similarity(Xt_f[:500], Xte_f[:500]).mean()
    sim_val_test = cosine_similarity(Xv_f[:500], Xte_f[:500]).mean()

    print("\nAverage Cosine Similarity")
    print("Train-Val :", sim_train_val)
    print("Train-Test:", sim_train_test)
    print("Val-Test  :", sim_val_test)


similarity_analysis()

def class_embedding_stats():

    def class_mean(X, y, label):
        return X[y == label].mean()

    stats = {
        "train_allergen": class_mean(X_train, y_train, 1),
        "train_non_allergen": class_mean(X_train, y_train, 0),
        "val_allergen": class_mean(X_val, y_val, 1),
        "val_non_allergen": class_mean(X_val, y_val, 0),
    }

    print("\nClass embedding statistics")
    for k, v in stats.items():
        print(k, ":", v)


class_embedding_stats()

print("\nDataset analysis finished")