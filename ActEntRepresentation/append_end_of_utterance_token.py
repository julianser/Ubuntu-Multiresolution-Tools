# Helper script

import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dialogues", type=str, default="", help="Input dialogues")
    parser.add_argument("output_dialogues", type=str, default="", help="Output dialogues with end-of-utterance token appended")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    eou_token = '__eou__'

    lines = open(args.input_dialogues, 'r').readlines()
    for lineidx, line in enumerate(lines):
        if line.strip().split()[-1] != eou_token:
            line = f'{line.strip()} {eou_token}' + '\n'
            lines[lineidx] = line.replace('  ', ' ')




    with open(args.output_dialogues, 'w') as out:
        for line in lines:
            out.write(line)
