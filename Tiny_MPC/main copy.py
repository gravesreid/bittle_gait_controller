
import argparse
parser = argparse.ArgumentParser(description="Tiny_MPC")
parser.add_argument("skill", type=int, help="1-bk \n 2-wkf \n 3-balance \n 4-wkf_matched_sequence")
parser.add_argument("improve_reference_gait", action="store_true", help="0 - Improve Reference Gait \n 1 - Do not improve reference gait")
args = parser.parse_args()

print("Number:", args.number)
if args.verbose:
    print("Verbose mode is on")