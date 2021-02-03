# Main script to run the graphing code

import os

import matplotlib.pyplot as plt

import datastructures

datastructures.SETTINGS  # reference to prevent optimise away
# dataDir_checkpoints = "/rds/user/hrjh2/hpc-work/checkpoints/"
dataDir_checkpoints = "/home/harrison/Documents/project/mount/rds/user/hrjh2/hpc-work/checkpoints/"

def loadAFile():
    while True:
        print("Select a file to load:")
        files = [f for f in os.listdir(dataDir_checkpoints) if os.path.isfile(os.path.join(dataDir_checkpoints, f))]
        fileNumMap = {}
        for file in files:
            if file.endswith(".evals"):
                end = file.split("_")[-1]
                rest = "_".join(file.split("_")[:-1])
                end = int(end[:-6])  # just the number
                if rest in fileNumMap.keys():
                    if end > fileNumMap[rest]:
                        fileNumMap[rest] = end
                    # else less than
                else:
                    fileNumMap[rest] = end
        fileShortnameMap = {}
        for short, num in fileNumMap.items():
            fileShortnameMap[short] = short+"_"+str(num)+".evals"
        for idx, short in enumerate(fileShortnameMap.keys()):
            print(f"{idx}) {short}")
        selection = input()
        short, file = list(fileShortnameMap.items())[int(selection)]
        print(f"Loading {short}({file})...")
        return file


def XYFromEvals(evals: datastructures.EvalHistory, valx, valy):
    x = []
    y = []
    for eval in evals.metrics:
        if valx == "step":
            x.append(eval.step)
        if valy == "acc":
            y.append(eval.accuracy)
        elif valy == "max":
            y.append(eval.accuracy_possible)
    return x, y


#Main loop
while True:
    # 1) Load 1st file
    file1Name = loadAFile()
    # 2) Load 2nd file
    file2Name = loadAFile()
    # 3) load as evals
    evals1 = datastructures.EvalHistory.load(dataDir_checkpoints + file1Name)
    evals2 = datastructures.EvalHistory.load(dataDir_checkpoints + file2Name)
    print("e1", evals1.metrics)
    # 4) Present comparison options
    print("Select comparison option:")
    print("0) Step-based | ratio")
    selection = input()
    # 5) Graph
    if selection == "0":
        print("Comparing...")
        x1, y1 = XYFromEvals(evals1, "step", "acc")
        x2, y2 = XYFromEvals(evals2, "step", "acc")
        x3, y3 = XYFromEvals(evals1, "step", "max")

        plt.figure()

        l1 = plt.plot(x1, y1, "r-", label=file1Name)[0]
        l2 = plt.plot(x2, y2, "g-", label=file2Name)[0]
        l3 = plt.plot(x3, y3, "k--", label="MAX")[0]
        plt.xlabel("Step")
        plt.ylabel("Accuracy")
        plt.ylim(0, 1)
        plt.legend(handles=[l1, l2, l3])
        plt.show()
