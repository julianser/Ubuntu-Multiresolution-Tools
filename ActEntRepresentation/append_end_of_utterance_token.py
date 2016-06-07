# Helper script

import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dialogues", type=str, default="", help="Input dialogues")
    parser.add_argument("output_dialogues", type=str, default="", help="Output dialogues with end-of-utterance token appended")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    eou_token = '__eou__'

    lines = open(args.input_dialogues, 'r').readlines()
    for lineidx, line in enumerate(lines):
        if line.strip().split()[-1] != eou_token:
            line = line.strip() + ' ' + eou_token + '\n'
            lines[lineidx] = line.replace('  ', ' ')


    

    out = open(args.output_dialogues, 'w')
    for line in lines:
        out.write(line)
    out.close()
