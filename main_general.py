# Main pipeline to run the paper, but for HPC usage
import sys
from hyperparameters import SETTINGS, APPLYBUNDLE_hpc, APPLYBUNDLE_mentNorm, APPLYBUNDLE_relNorm, \
    APPLYBUNDLE_mentNormK1, APPLYBUNDLE_mentNormNoPad, APPLYBUNDLE_paper
import argparse
#TODO researched Argparse - from the standard lib

parser = argparse.ArgumentParser()
parser.add_argument(
    "-n","--name",
    type=str,
    help="Name of the save file",
    default="save_default"
)
parser.add_argument(
    "-b","--bundles","--bundle",
    nargs="*",
    help="Name of bundles to invoke",
    default=[]
)
args = parser.parse_args()
SETTINGS.saveName = args.name
normBundled = None
def normCheck(str):
    global normBundled
    if normBundled is not None:
        print(f"Tried to apply normalisation bundles '{str}' and '{normBundled}'")
        print("Cannot have 2 normalisation bundles.")
        quit(1)
    else:
        normBundled = str
for bundleName in args.bundles:
    if bundleName == "hpc":
        APPLYBUNDLE_hpc(SETTINGS)
    elif bundleName == "ment":
        APPLYBUNDLE_mentNorm(SETTINGS)
        normCheck("ment")
    elif bundleName == "rel":
        APPLYBUNDLE_relNorm(SETTINGS)
        normCheck("rel")
    elif bundleName == "mentk1":
        APPLYBUNDLE_mentNormK1(SETTINGS)
        normCheck("mentk1")
    elif bundleName == "mentNoPad":
        APPLYBUNDLE_mentNormNoPad(SETTINGS)
        normCheck("mentNoPad")
    elif bundleName == "paper":
        APPLYBUNDLE_paper(SETTINGS)
    else:
        print(f"Unknown bundle name '{bundleName}'")
        quit(1)

print("SETTINGS:",SETTINGS)
import main
print("Results:")
main.model.evals.print()
