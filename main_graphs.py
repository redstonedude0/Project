# Main script to run the graphing code

import os

import matplotlib.pyplot as plt

import datastructures

datastructures.SETTINGS  # reference to prevent optimise away
# dataDir_checkpoints = "/rds/user/hrjh2/hpc-work/checkpoints/"
dataDir_checkpoints = "/home/harrison/Documents/project/mount2/checkpoints/"

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

def loadARun():
    runNumMap = computeRunMaps()
    for idx, name in enumerate(runNumMap.keys()):
        length = len(runNumMap[name])
        print(f"{idx}) {name} ({length})")
    print("Selection:")
    selection = input()
    name, idxMap = list(runNumMap.items())[int(selection)]
    print(f"Loading {name}...")
    return (name,idxMap)

def computeRunMaps():
    test_types = ["blind0", "blind1", "blind2",
                  "blind3", "blind4", "blind5", "blind6",
                  "blind7", "blind", ""]
    runnames = ["auto_"+test_type for test_type in test_types]
    runnames.append("mrna")
    print("Select a run to load:")
    files = [f for f in os.listdir(dataDir_checkpoints) if os.path.isfile(os.path.join(dataDir_checkpoints, f))]
    runNumMap = {} #runName->idx->num
    for file in files:
        if file.endswith(".evals"):#evaluation file
            is_run = None
            for runname in runnames:
                if file.startswith(runname):
                    is_run = runname
                    break
            if is_run is not None:#is actually a run
                end = file.split("_")[-1]
                idx = int(file.split("_")[-2:-1][0])
                num = int(end[:-6])  # just the number (.evals removed)
                if is_run in runNumMap.keys():
                    if idx in runNumMap[is_run].keys():
                        if num > runNumMap[is_run][idx]:
                            runNumMap[is_run][idx] = num
                        # else less than
                    else:
                        runNumMap[is_run][idx] = num
                else:
                    runNumMap[is_run] = {}
                    runNumMap[is_run][idx] = num
    return runNumMap

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


def compare2():
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
        print("0) ratio | step")
        print("1) loss  | step")
        print("2) step  | time")
        selection = input()
        # 5) Graph
        if selection == "0":
            print("Comparing...")
            x1, y1 = XYFromEvals(evals1, "step", "acc")
            x2, y2 = XYFromEvals(evals2, "step", "acc")
            x3, y3 = XYFromEvals(evals1, "step", "max")

            plt.figure()
            file1Name = "Mine (Initial) (ment-norm)"
            file2Name = "Mine (Fully Adjusted) (ment-norm)"

            l1 = plt.plot(x1, y1, "r-", label=file1Name)[0]
            l2 = plt.plot(x2, y2, "g-", label=file2Name)[0]
    #        l3 = plt.plot(x3, y3, "k--", label="MAX")[0]
            plt.xlabel("Step")
            plt.ylabel("Micro-F1")
            plt.ylim(0, 1)
            plt.legend(handles=[l1,l2])
    #        plt.legend(handles=[l1, l2, l3])
            plt.show()
        elif selection == "1":
            print("Comparing...")
            x1, y1 = XYFromEvals(evals1, "step", "loss")
            x2, y2 = XYFromEvals(evals2, "step", "loss")

            plt.figure()

            l1 = plt.plot(x1, y1, "r-", label=file1Name)[0]
            l2 = plt.plot(x2, y2, "g-", label=file2Name)[0]
            plt.xlabel("Step")
            plt.ylabel("Loss")
            plt.ylim(0, 1)
            plt.legend(handles=[l1, l2])
            plt.show()
        elif selection == "2":
            print("Comparing...")
            x1, y1 = XYFromEvals(evals1, "time", "step")
            x2, y2 = XYFromEvals(evals2, "time", "step")

            plt.figure()

            l1 = plt.plot(x1, y1, "r-", label=file1Name)[0]
            l2 = plt.plot(x2, y2, "g-", label=file2Name)[0]
            plt.xlabel("Time")
            plt.ylabel("Step")
            plt.ylim(0, 1)
            plt.legend(handles=[l1, l2])
            plt.show()
        max1,max2 = 0,0
        for eval1,eval2 in zip(evals1.metrics,evals2.metrics):
            max1 = max(max1,eval1.accuracy)
            max2 = max(max2,eval2.accuracy)
        print("MAX1,2",max1,max2)

def computeAvgEval(evals):
    avgHistory = datastructures.EvalHistory()
    avgHistory.metrics = []
    count = len(evals)
    length = 9999
    for eval in evals:
        length = min(length,len(eval.metrics))
    for idx in range(0,length):
        accuracySum = 0
        step = evals[0].metrics[idx].step
        for eval in evals:
            accuracySum += eval.metrics[idx].accuracy
        evalmetrics = datastructures.EvaluationMetrics()
        evalmetrics.accuracy = accuracySum / count
        evalmetrics.accuracy_possible = 0
        evalmetrics.loss = 0
        evalmetrics.time = 0
        evalmetrics.step = step
        avgHistory.metrics.append(evalmetrics)
    return avgHistory

def compare_more():
    # Main loop
    while True:
        # Load a run
        name,idxMap = loadARun()
        # load as evals
        evals = []
        for idx,num in idxMap.items():
            evals.append(datastructures.EvalHistory.load(dataDir_checkpoints + name+"_"+str(idx)+"_"+str(num)+".evals"))
        avgEval = computeAvgEval(evals)
        print("Comparing...")
        plt.figure()
        handles = []
        idx = 1
        length = len(avgEval.metrics)
        for eval in evals:
            x1, y1 = XYFromEvals(eval, "step", "acc")
            print(name,":",len(x1),max(y1))
            x1 = x1[:length]
            y1 = y1[:length]
            l1 = plt.plot(x1, y1,"r-",label=f"Mine ({name}) Runs 1-5", linewidth=0.5)[0]
            idx += 1
        handles.append(l1)

        x1, y1 = XYFromEvals(avgEval, "step", "acc")
        print("AVG:", len(x1),max(y1))
        plotName = f"Mine ({name}) Mean"
        l1 = plt.plot(x1, y1, "g--", label=plotName,linewidth=2)[0]
        handles.append(l1)


        plt.xlabel("Step")
        plt.ylabel("Micro-F1")
        plt.ylim(0, 1)
        plt.xticks(range(0,length+1,2))
        plt.legend(handles=handles)
        #        plt.legend(handles=[l1, l2, l3])
        plt.show()

def round(float,fun,digits=4):
    float *= 10 ** digits
    return "{1:.{0}f}".format(digits,fun(float) / (10**digits))

def compute_stats():
    # Load runs
    runNumMap = computeRunMaps()
    for idx, name in enumerate(runNumMap.keys()):
        length = len(runNumMap[name])
        print(f"{idx}) {name} ({length})")
        idxMap = runNumMap[name]
        # load as evals
        evals = []
        for idx,num in idxMap.items():
            evals.append(datastructures.EvalHistory.load(dataDir_checkpoints + name+"_"+str(idx)+"_"+str(num)+".evals"))
        avgEval = computeAvgEval(evals)
        length = len(avgEval.metrics)
        lengths = []
        bestScores = []
        for eval in evals:
            lengths.append(len(eval.metrics))
            bestScores.append(max([m.accuracy for m in eval.metrics]))
        for length,bestScore in zip(lengths,bestScores):
            print("  ",length,bestScore)
        avgLen = sum(lengths)/len(lengths)
        avgScore = sum(bestScores)/len(bestScores)
        lenPM = max(lengths)-avgLen
        lenPM = max(lenPM,avgLen-min(lengths))
        scorePM = max(bestScores)-avgScore
        scorePM = max(scorePM,avgScore-min(bestScores))
        print("  AVG LEN:",avgLen,lenPM)
        print("  AVG SCR:",avgScore,scorePM)
        import math
        print(round(avgScore,math.floor),"\\pm",round(scorePM,math.ceil)," LEN:",
              round(avgLen,math.ceil),"\\pm",round(lenPM,math.ceil))

compare2()
#compare_more()
#compute_stats()