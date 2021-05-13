# Main script to run the graphing code

import os
import math

import matplotlib.pyplot as plt

import datastructures

datastructures.SETTINGS  # reference to prevent optimise away
# data_dir_checkpoints = "/rds/user/hrjh2/hpc-work/checkpoints/"
data_dir_checkpoints = "/home/harrison/Documents/project/mount2/checkpoints/"

def load_a_file():
    while True:
        print("Select a file to load:")
        files = [f for f in os.listdir(data_dir_checkpoints) if os.path.isfile(os.path.join(data_dir_checkpoints, f))]
        file_num_map = {}
        for file in files:
            if file.endswith(".evals"):
                end = file.split("_")[-1]
                rest = "_".join(file.split("_")[:-1])
                end = int(end[:-6])  # just the number
                if rest in file_num_map.keys():
                    if end > file_num_map[rest]:
                        file_num_map[rest] = end
                    # else less than
                else:
                    file_num_map[rest] = end
        file_shortname_map = {}
        for short, num in file_num_map.items():
            file_shortname_map[short] = short+"_"+str(num)+".evals"
        for idx, short in enumerate(file_shortname_map.keys()):
            print(f"{idx}) {short}")
        selection = input()
        short, file = list(file_shortname_map.items())[int(selection)]
        print(f"Loading {short}({file})...")
        return file

def load_a_run():
    run_num_map = compute_run_maps()
    for idx, name in enumerate(run_num_map.keys()):
        length = len(run_num_map[name])
        print(f"{idx}) {name} ({length})")
    print("Selection:")
    selection = input()
    name, idx_map = list(run_num_map.items())[int(selection)]
    print(f"Loading {name}...")
    return (name,idx_map)

def compute_run_maps():
    test_types = ["blind0", "blind1", "blind2",
                  "blind3", "blind4", "blind5", "blind6",
                  "blind7", "blind", ""]
    run_names = ["auto_"+test_type for test_type in test_types]
    run_names.append("mrna")
    print("Select a run to load:")
    files = [f for f in os.listdir(data_dir_checkpoints) if os.path.isfile(os.path.join(data_dir_checkpoints, f))]
    run_num_map = {} #runName->idx->num
    for file in files:
        if file.endswith(".evals"):#evaluation file
            is_run = None
            for run_name in run_names:
                if file.startswith(run_name):
                    is_run = run_name
                    break
            if is_run is not None:#is actually a run
                end = file.split("_")[-1]
                idx = int(file.split("_")[-2:-1][0])
                num = int(end[:-6])  # just the number (.evals removed)
                if is_run in run_num_map.keys():
                    if idx in run_num_map[is_run].keys():
                        if num > run_num_map[is_run][idx]:
                            run_num_map[is_run][idx] = num
                        # else less than
                    else:
                        run_num_map[is_run][idx] = num
                else:
                    run_num_map[is_run] = {}
                    run_num_map[is_run][idx] = num
    return run_num_map

def xy_from_evals(evals: datastructures.EvalHistory, val_x, val_y):
    x = []
    y = []
    for eval in evals.metrics:
        if val_x == "step":
            x.append(eval.step)
        if val_y == "acc":
            y.append(eval.accuracy)
        elif val_y == "max":
            y.append(eval.accuracy_possible)
    return x, y


def compare_two():
    #Main loop
    while True:
        # 1) Load 1st file
        file1_name = load_a_file()
        # 2) Load 2nd file
        file2_name = load_a_file()
        # 3) load as evals
        evals1 = datastructures.EvalHistory.load(data_dir_checkpoints + file1_name)
        evals2 = datastructures.EvalHistory.load(data_dir_checkpoints + file2_name)
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
            x1, y1 = xy_from_evals(evals1, "step", "acc")
            x2, y2 = xy_from_evals(evals2, "step", "acc")
            x3, y3 = xy_from_evals(evals1, "step", "max")

            plt.figure()
            file1_name = "Mine (Initial) (ment-norm)"
            file2_name = "Mine (Fully Adjusted) (ment-norm)"

            l1 = plt.plot(x1, y1, "r-", label=file1_name)[0]
            l2 = plt.plot(x2, y2, "g-", label=file2_name)[0]
    #        l3 = plt.plot(x3, y3, "k--", label="MAX")[0]
            plt.xlabel("Step")
            plt.ylabel("Micro-F1")
            plt.ylim(0, 1)
            plt.legend(handles=[l1,l2])
    #        plt.legend(handles=[l1, l2, l3])
            plt.show()
        elif selection == "1":
            print("Comparing...")
            x1, y1 = xy_from_evals(evals1, "step", "loss")
            x2, y2 = xy_from_evals(evals2, "step", "loss")

            plt.figure()

            l1 = plt.plot(x1, y1, "r-", label=file1_name)[0]
            l2 = plt.plot(x2, y2, "g-", label=file2_name)[0]
            plt.xlabel("Step")
            plt.ylabel("Loss")
            plt.ylim(0, 1)
            plt.legend(handles=[l1, l2])
            plt.show()
        elif selection == "2":
            print("Comparing...")
            x1, y1 = xy_from_evals(evals1, "time", "step")
            x2, y2 = xy_from_evals(evals2, "time", "step")

            plt.figure()

            l1 = plt.plot(x1, y1, "r-", label=file1_name)[0]
            l2 = plt.plot(x2, y2, "g-", label=file2_name)[0]
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

def compute_avg_eval(evals):
    avg_history = datastructures.EvalHistory()
    avg_history.metrics = []
    count = len(evals)
    length = 9999
    for eval in evals:
        length = min(length,len(eval.metrics))
    for idx in range(0,length):
        accuracy_sum = 0
        step = evals[0].metrics[idx].step
        for eval in evals:
            accuracy_sum += eval.metrics[idx].accuracy
        eval_metrics = datastructures.EvaluationMetrics()
        eval_metrics.accuracy = accuracy_sum / count
        eval_metrics.accuracy_possible = 0
        eval_metrics.loss = 0
        eval_metrics.time = 0
        eval_metrics.step = step
        avg_history.metrics.append(eval_metrics)
    return avg_history

def compare_more():
    # Main loop
    while True:
        # Load a run
        name,idx_map = load_a_run()
        # load as evals
        evals = []
        for idx,num in idx_map.items():
            evals.append(datastructures.EvalHistory.load(data_dir_checkpoints + name + "_" + str(idx) + "_" + str(num) + ".evals"))
        avg_eval = compute_avg_eval(evals)
        print("Comparing...")
#        name = "Final"
        plt.figure()
        handles = []
        idx = 1
        length = len(avg_eval.metrics)
        for eval in evals:
            x1, y1 = xy_from_evals(eval, "step", "acc")
            print(name,":",len(x1),max(y1))
            x1 = x1[:length]
            y1 = y1[:length]
            l1 = plt.plot(x1, y1,"r-",label=f"Mine ({name}) Runs 1-5", linewidth=0.5)[0]
            idx += 1
        handles.append(l1)

        x1, y1 = xy_from_evals(avg_eval, "step", "acc")
        print("AVG:", len(x1),max(y1))
        plot_name = f"Mine ({name}) Mean"
        l1 = plt.plot(x1, y1, "g--", label=plot_name,linewidth=2)[0]
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
    run_num_map = compute_run_maps()
    for idx, name in enumerate(run_num_map.keys()):
        length = len(run_num_map[name])
        print(f"{idx}) {name} ({length})")
        idx_map = run_num_map[name]
        # load as evals
        evals = []
        for idx,num in idx_map.items():
            evals.append(datastructures.EvalHistory.load(data_dir_checkpoints + name + "_" + str(idx) + "_" + str(num) + ".evals"))
        avg_eval = compute_avg_eval(evals)
        length = len(avg_eval.metrics)
        lengths = []
        best_scores = []
        steps = []
        for eval in evals:
            lengths.append(len(eval.metrics))
            best_scores.append(max([m.accuracy for m in eval.metrics]))
            last_acc = -1
            for m in eval.metrics:
                acc = m.accuracy
                if last_acc != -1 and m.step <= 10:
                    steps.append(abs(last_acc-acc))
                last_acc = acc
        for length,bestScore in zip(lengths,best_scores):
            print("  ",length,bestScore)
        avg_len = sum(lengths)/len(lengths)
        avg_score = sum(best_scores)/len(best_scores)
        avg_step = sum(steps)/len(steps)
        len_pm = max(lengths)-avg_len
        len_pm = max(len_pm,avg_len-min(lengths))
        score_pm = max(best_scores)-avg_score
        score_pm = max(score_pm,avg_score-min(best_scores))
        print("  AVG LEN:",avg_len,len_pm)
        print("  AVG SCR:",avg_score,score_pm)
        print("  AVG STEP/10:",avg_step)
        print(round(avg_score,math.floor),"\\pm",round(score_pm,math.ceil)," LEN:",
              round(avg_len,math.ceil),"\\pm",round(len_pm,math.ceil))

#compare_two()
#compare_more()
compute_stats()