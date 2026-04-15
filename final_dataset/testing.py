import json

with open("train.json", "r") as f:
    train_data = json.load(f)

print("Number of train samples:", len(train_data))

with open("test.json", "r") as f:
    train_data = json.load(f)

print("Number of test samples:", len(train_data))