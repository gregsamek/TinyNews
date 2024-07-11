import torch

if __name__ == "__main__":
    for s in ("train", "validation"):
        txt8 = []
        with open(f"training_data/{s}_8b.txt", 'r') as f:
            for line in f:
                txt8.append(line)

        txt70 = []
        with open(f"training_data/{s}_70b.txt", 'r') as f:
            for line in f:
                txt70.append(line)

        txt = txt8 + txt70

        with open(f"training_data/{s}.txt", 'w') as f:
            for line in txt:
                f.write(line)
    