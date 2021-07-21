import json
import numpy as np
import pandas as pd
import os
import cv2

def load(path):
    with open(path) as f:
        return json.load(f)

def get_classes(path="data_json/names.csv", sep=","):
    classes = pd.read_csv(path, sep=sep).iloc[:,1]
    return classes.tolist()

def make_dir(path):
    dirs = path.split(sep="/")
    for i in range(len(dirs)):
        if not os.path.exists("/".join(dirs[:i+1])):
            os.mkdir("/".join(dirs[:i+1]))

def convert():
    X = {x: np.array(load(f"data_json/{x}_data.json"), dtype="uint8") for x in ["train", "val"]}
    y = {x: np.array(load(f"data_json/{x}_labels.json"))[1:] for x in ["train", "val"]}

    for phase in ["train", "val"]:
        y[phase] = np.array(y[phase], dtype="uint8")

    classes = get_classes()

    for phase in ["train", "val"]:
        for i, img in enumerate(X[phase]):
            full_dir = f"data/{phase}/{classes[y[phase][i]]}"
            make_dir(full_dir)
            cv2.imwrite(full_dir + f"/{i}.png", img)

    