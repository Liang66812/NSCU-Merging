import torch
from torch.utils.data import DataLoader, Dataset
from transformers import ViTForImageClassification, ViTFeatureExtractor, Trainer, TrainingArguments, ViTImageProcessor
import os
from PIL import Image
from sklearn.metrics import accuracy_score
import numpy as np
import pickle
import pyarrow.parquet as pq
import io
import pandas as pd 
from peft import LoraConfig, TaskType, get_peft_model,PeftModel
from sklearn.model_selection import train_test_split

feature_extractor = ViTImageProcessor.from_pretrained("/mnt/data_B/lyf/vit-base-patch16-224")

class CustomTensorDataset(Dataset):
    def __init__(self, tensors, labels, mode= "PNG_code", transform=None): #"JPG"  Tensor
        self.tensors = tensors
        self.labels = labels
        self.transform = transform
        self.mode = mode
        assert len(self.tensors) == len(self.labels), "张量数量和标签数量必须相同"

    def __len__(self):
        return len(self.tensors)

    def __getitem__(self, idx):
        if self.mode == "PNG_code":
            image = Image.open(io.BytesIO(self.tensors[idx])).convert('RGB')
        elif self.mode == "JPG":
            image = Image.open(self.tensors[idx]).convert('RGB')
        else:
            image = self.tensors[idx]
        encoding = feature_extractor(images=image, return_tensors="pt")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return {"pixel_values": encoding["pixel_values"].squeeze(), "labels": torch.tensor(label)}
    
def get_mnist_data():
    df_train = pq.read_table('/mnt/data_B/lyf/vision_classify_ds/mnist/mnist/train-00000-of-00001.parquet').to_pandas()
    df_test = pq.read_table('/mnt/data_B/lyf/vision_classify_ds/mnist/mnist/test-00000-of-00001.parquet').to_pandas()
    length = 2000
    train_list = [df_train["image"][i]["bytes"] for i in range(len(df_train[:length]))]
    test_list = [df_test["image"][i]["bytes"] for i in range(len(df_train[:length]))]

    train_dataset = CustomTensorDataset(train_list, list(df_train["label"][:length]))
    test_dataset = CustomTensorDataset(test_list, list(df_test["label"][:length]))
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    return train_dataset, test_dataset

def get_resis_data():
    df_train = pq.read_table('/mnt/data_B/lyf/vision_classify_ds/resisc45/data/train-00000-of-00001.parquet').to_pandas()
    df_test = pq.read_table('/mnt/data_B/lyf/vision_classify_ds/resisc45/data/test-00000-of-00001.parquet').to_pandas()
    length = 10000
    train_list = [df_train["image"][i]["bytes"] for i in range(len(df_train[:length]))]
    test_list = [df_test["image"][i]["bytes"] for i in range(len(df_test[:length]))]

    train_dataset = CustomTensorDataset(train_list, list(df_train["label"][:length]))
    test_dataset = CustomTensorDataset(test_list, list(df_test["label"][:length]))
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    return train_dataset, test_dataset


def get_EuroSAT_data():
    image_list = []
    label_list = []
    for i, filename in enumerate(os.listdir("/mnt/data_B/lyf/vision_classify_ds/EuroSAT/raw/2750")):
        # print(i, filename)
        for filename_ in os.listdir(os.path.join("/mnt/data_B/lyf/vision_classify_ds/EuroSAT/raw/2750",filename)):
            # print(os.path.join("/mnt/data_B/lyf/vision_classify_ds/EuroSAT/raw/2750",filename,filename_))
            image_list.append(os.path.join("/mnt/data_B/lyf/vision_classify_ds/EuroSAT/raw/2750",filename,filename_))
            label_list.append(i)
    train_list, test_list, train_labels, test_labels = train_test_split(
        image_list, label_list, test_size=0.2, random_state=42
    )
    train_dataset = CustomTensorDataset(train_list, train_labels, mode="JPG")
    test_dataset = CustomTensorDataset(test_list, test_labels,mode="JPG")
    return train_dataset, test_dataset


def load_cifar10_batch(file_path):
    with open(file_path, 'rb') as f:
        data_dict = pickle.load(f, encoding='bytes')  # Python3 必须加 encoding
        images = data_dict[b'data']                  # 图像数据: shape = [10000, 3072]
        labels = data_dict[b'labels']                # 标签: shape = [10000]

        # 将扁平的图像数据重塑为 [N, C, H, W] 格式
        images = images.reshape(-1, 3, 32, 32).astype(np.uint8)
        return images, np.array(labels)
    

def get_cifar10_data():
    images, labels = load_cifar10_batch("/mnt/data_B/lyf/vision_classify_ds/CIFAR-10/data/cifar-10-batches-py/data_batch_1")

    train_list, test_list, train_labels, test_labels = train_test_split(
        images, labels, test_size=0.2, random_state=42
    )

    train_dataset = CustomTensorDataset(train_list, train_labels, mode="Tensor")
    test_dataset = CustomTensorDataset(test_list, test_labels,mode="Tensor")
    return train_dataset, test_dataset


def get_car_data():
    df_train = pd.read_csv("/mnt/data_B/lyf/vision_classify_ds/carBrands50/train.csv")
    train_list = [os.path.join("/mnt/data_B/lyf/vision_classify_ds/carBrands50",i) for i in df_train["image:FILE"]]
    df_test = pd.read_csv("/mnt/data_B/lyf/vision_classify_ds/carBrands50/val.csv")
    test_list = [os.path.join("/mnt/data_B/lyf/vision_classify_ds/carBrands50",i) for i in df_test["image:FILE"]]
    train_dataset = CustomTensorDataset(train_list, list(df_train["category"]), mode="JPG")
    test_dataset = CustomTensorDataset(test_list, list(df_test["category"]), mode="JPG")
    return train_dataset, test_dataset


def get_fruits_data(): 
    df_train = pd.read_csv("/mnt/data_B/lyf/vision_classify_ds/fruits100/train.csv")
    train_list = [os.path.join("/mnt/data_B/lyf/vision_classify_ds/fruits100",i) for i in df_train["image:FILE"]]
    df_test = pd.read_csv("/mnt/data_B/lyf/vision_classify_ds/fruits100/val.csv")
    test_list = [os.path.join("/mnt/data_B/lyf/vision_classify_ds/fruits100",i) for i in df_test["image:FILE"]]
    train_dataset = CustomTensorDataset(train_list, list(df_train["category"]), mode="JPG")
    test_dataset = CustomTensorDataset(test_list, list(df_test["category"]), mode="JPG")
    return train_dataset, test_dataset

def get_GTSRB_data():
    images = []
    labels = []
    start_path = "/mnt/data_B/lyf/vision_classify_ds/GTSRB/Final_Training/Images/"
    for i, filename in enumerate(os.listdir(start_path)):
        if filename.startswith("0"):
            for filename_ in os.listdir(os.path.join(start_path,filename)):
                if filename_.endswith("csv"):
                    df = pd.read_csv(os.path.join(start_path,filename,filename_ ))
                    images += [os.path.join(start_path,filename,i.split(";")[0]) for i in list(df.iloc[:, 0])]
                    labels += [int(i.split(";")[-1]) for i in list(df.iloc[:, 0])]
    train_list, test_list, train_labels, test_labels = train_test_split(
            images, labels, test_size=0.85, random_state=42
        )
    train_dataset = CustomTensorDataset(train_list, train_labels, mode="JPG")
    test_dataset = CustomTensorDataset(test_list, test_labels,mode="JPG")
    return train_dataset, test_dataset

def get_DTD_data():
    image_list = []
    label_list = []
    start_path = "/mnt/data_B/lyf/vision_classify_ds/DTD/images"
    for i, filename in enumerate(os.listdir(start_path)):
        if not filename.endswith("Store"):
            for filename_ in os.listdir(os.path.join(start_path,filename)):
                if filename_.endswith("jpg"):
                    image_list.append(os.path.join(start_path,filename,filename_))
                    label_list.append(i)
    image_list.pop(3407)
    label_list.pop(3407)
    train_list, test_list, train_labels, test_labels = train_test_split(
        image_list, label_list, test_size=0.2, random_state=42
    )
    train_dataset = CustomTensorDataset(train_list, train_labels, mode="JPG")
    test_dataset = CustomTensorDataset(test_list, test_labels,mode="JPG")
    return train_dataset, test_dataset

def get_grabage_data():
    train_list = []
    train_labels = []
    start_path = "/mnt/data_B/lyf/vision_classify_ds/df_grabage/train"
    for i, filename in enumerate(os.listdir(start_path)):
        if filename.startswith("0") or filename.startswith("1"):
            for filename_ in os.listdir(os.path.join(start_path,filename)):
                if filename_.endswith("png"):
                    train_list.append(os.path.join(start_path,filename,filename_))
                    train_labels.append(i)
    test_list = []
    test_labels = []
    start_path = "/mnt/data_B/lyf/vision_classify_ds/df_grabage/val"
    for i, filename in enumerate(os.listdir(start_path)):
        if filename.startswith("0") or filename.startswith("1"):
            for filename_ in os.listdir(os.path.join(start_path,filename)):
                if filename_.endswith("png"):
                    test_list.append(os.path.join(start_path,filename,filename_))
                    test_labels.append(i)
    
    train_dataset = CustomTensorDataset(train_list, train_labels, mode="JPG")
    test_dataset = CustomTensorDataset(test_list, test_labels,mode="JPG")
    return train_dataset, test_dataset

def get_plants_data():
    df = pd.read_csv('/mnt/data_B/lyf/vision_classify_ds/plants/plants_train.csv')
    image_list = [os.path.join("/mnt/data_B/lyf/vision_classify_ds/plants", name) for name in df["Image"]]
    label_list = [int(label) for label in df["CATEGORY"]]
    train_list, test_list, train_labels, test_labels = train_test_split(
        image_list, label_list, test_size=0.7, random_state=42
    )
    train_dataset = CustomTensorDataset(train_list, train_labels, mode="JPG")
    test_dataset = CustomTensorDataset(test_list, test_labels,mode="JPG")
    return train_dataset, test_dataset