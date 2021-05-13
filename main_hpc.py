# Main pipeline to run the paper, but for HPC usage
import sys
from hyperparameters import SETTINGS, apply_bundle_hpc, apply_bundle_ment_norm,apply_bundle_rel_norm,apply_bundle_ment_norm_k1,apply_bundle_ment_norm_no_pad
import argparse
#TODO researched Argparse - from the standard lib
print("DEPRECATED")
quit(0)


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
SETTINGS.save_name = args.name
norm_bundled = None
def norm_check(str):
    global norm_bundled
    if norm_bundled is not None:
        print(f"Tried to apply normalisation bundles '{str}' and '{norm_bundled}'")
        print("Cannot have 2 normalisation bundles.")
        quit(1)
    else:
        normBundled = str
for bundle_name in args.bundles:
    if bundle_name == "hpc":
        apply_bundle_hpc(SETTINGS)
    elif bundle_name == "ment":
        apply_bundle_ment_norm(SETTINGS)
        norm_check("ment")
    elif bundle_name == "rel":
        apply_bundle_rel_norm(SETTINGS)
        norm_check("rel")
    elif bundle_name == "mentk1":
        apply_bundle_ment_norm_k1(SETTINGS)
        norm_check("mentk1")
    elif bundle_name == "mentNoPad":
        apply_bundle_ment_norm_no_pad(SETTINGS)
        norm_check("mentNoPad")
    else:
        print(f"Unknown bundle name '{bundle_name}'")
        quit(1)

import main

print("HPC Results:")
main.model.evals.print()
