TEMPLATE_FILENAME = "../slurm_auto_template.wilkes2"
TARGET_FILENAME = "../genslurm"
TEMPLATE_PLACEHOLDER = "%INSERT_OPTIONS_LINE%"

def gen_myoptions(name,test_type):
    return f"options=\"/home/hrjh2/code/main_general.py --name {name} --bundles paper rel hpc {test_type} >/home/hrjh2/out_{name}.txt 2>/home/hrjh2/err_{name}.txt\""

def gen_theiroptions(name):
    return f"options=\"-u -m nel.main --mode train --n_rels 3 --mulrel_type ment-norm --model_path {name} >/home/hrjh2/out_{name}.txt 2>/home/hrjh2/err_{name}.txt\""

def make(name,optionsline):
    with open(TEMPLATE_FILENAME,"r") as f_source:
        with open(TARGET_FILENAME+f"_{name}","w") as f_target:
            for line in f_source:
                if line.startswith(TEMPLATE_PLACEHOLDER):
                    line = line.replace(TEMPLATE_PLACEHOLDER,optionsline)
                f_target.write(line)

def makefiles():
    test_types = ["blind","blind0","blind1","blind2",
             "blind3","blind4","blind5","blind6",
             "blind7",""]
    for test_type in test_types:
        for i in range(0,5):
            name = f"autorel_{test_type}_{i}"
            optionsline = gen_myoptions(name,test_type)
            make(name,optionsline)
    for i in range(0,5):
        name = f"mrna_{i}"
        optionsline = gen_theiroptions(name)
        make(name, optionsline)

makefiles()

