import os
from tqdm import tqdm
import json
import logging
import argparse
from utils.utils import load_source
from utils.llm import get_llm_result
from utils.prompt import get_prompt


ra_dict = {
    'none': 'none',
    'sparse': {'sparse_ctxs': 10},
    'dense': {'dense_ctxs': 10},
    'chatgpt': {'gen_ctxs': 100},
    'sparse+dense': {'sparse_ctxs': 5, 'dense_ctxs': 5},
    'gold': {'gold_ctxs': 10},
    'strong': {'strong_ctxs': 10},
    'weak': {'weak_ctxs': 10},
    'rand': {'rand_ctxs': 10},
}


def get_args():
    parser = argparse.ArgumentParser()
 
    parser.add_argument('--source', type=str, default='source/nq.json')
    parser.add_argument('--usechat', action='store_true')
    parser.add_argument('--type', type=str, choices=['qa', 'prior', 'post', 'generate'], default='qa')
    parser.add_argument('--ra', type=str, default="dense")
    parser.add_argument('--outfile', type=str, default='qa/chatgpt-nq-dense.json')
    
    args = parser.parse_args()
    args.ra = ra_dict[args.ra]

    return args


def main():

    args = get_args()
    begin = 0
    if os.path.exists(args.outfile):
        outfile = open(args.outfile, 'r', encoding='utf-8')
        for line in outfile.readlines():
            if line != "":
                begin += 1
        outfile.close()
        outfile = open(args.outfile, 'a', encoding='utf-8')
    else:
        outfile = open(args.outfile, 'w', encoding='utf-8')

    all_data = load_source(args.source)
    num_output = 0

    try:
        for sample in tqdm(all_data[begin:], desc="Filename: %s" % args.outfile):
            
            prompt = get_prompt(sample, args)
            sample = get_llm_result(prompt, args.usechat, sample, args.type)

            outfile.write(json.dumps(sample) + "\n")
            num_output += 1
    except Exception as e:
        logging.exception(e)
        
    finally:
        print(args.outfile, " has output %d line(s)." % num_output)
        outfile.close()


if __name__ == '__main__':
    main()
