import pickle

with open("data.pickle", "rb") as f:
    data = pickle.load(f)
    for row in data["data"]:
        for num in row:
            print(num)
