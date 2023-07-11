import json
from tqdm import tqdm
import argparse
import json
from tqdm import tqdm
from random import randint, seed
from utils.utils import has_answer


source_dic = {
    'nq': {
        'dense': 'source/nq-rocketqav2-top100',
        'sparse': 'source/nq-bm25-top1000',
        'qa': 'source/nq-qa',
        'outfile': 'source/nq.json',
    },
    'tq': {
        'dense': 'source/tq-rocketqav2-top100',
        'sparse': 'source/tq-bm25-top1000',
        'qa': 'source/tq-qa',
        'outfile': 'source/tq.json',
    },
    'hq': {
        'dense': 'source/hq-rocketqav2-top100',
        'sparse': 'source/hq-bm25-top1000',
        'qa': 'source/hq-qa',
        'outfile': 'source/hq.json',
    },
}


def load_ql(res_dir, top=1000):
    file = open(res_dir, 'r', encoding='utf-8')
    i = 0
    dl = []
    ql = []
    for line in tqdm(file.readlines()):
        line = line.split()
        i += 1
        dl.append(int(line[1]))
        if i == top:
            ql.append(dl)
            dl = []
            i = 0
    file.close()
    return ql


def get_dall(ql, topk, d_all=set()):
    if topk == 0:
        topk = len(ql[0])
    # 用于生成doc字典，返回所有需要的keys
    for cands in tqdm(ql):
        for did in cands[:topk]:
            d_all.add(did)
    return d_all


def read_doc(doc_dir, d_all):
    # 返回doc字典
    doc = {}
    file = open(doc_dir, 'r', encoding='utf-8')
    for line in tqdm(file.readlines()):
        line = line.split('\t')
        if int(line[0]) in d_all:
            doc[int(line[0])] = line[1]
    file.close()
    return doc


def get_llm(file):
    f = open(file, 'r', encoding='utf-8')
    p = []
    for line in f.readlines():
        line = json.loads(line)["predict"].replace("\n", " ")
        if line[0] == '?':
            line = line[1:]
        line = line.strip()
        p.append(line)
    return p

# para = get_r(ql)


def get_qa(filepath):
    file = open(filepath, 'r', encoding='utf-8')
    query, ans = [], []
    for line in file.readlines():
        line = line.strip('\n').split('\t')
        query.append(line[0])
        ans.append(line[1:])
    return query, ans


def gettxt(t, d):
    return " Title: " + t.strip() +  " Content: " + d.strip()


def get_args():
    parser = argparse.ArgumentParser()
 
    parser.add_argument('--dataset', '-d', type=str, choices=['nq', 'tq', 'hq'], default='nq')
    
    args = parser.parse_args()

    return args


def main():

    args = get_args()
    seed(114514)
    drand = set()
    dr = []
    for _ in range(361000):
        x = randint(0, 21015323)
        drand.add(x)
        dr.append(x)

    ql = {
        "bm25": load_ql(res_dir=source_dic[args.dataset]['sparse'], top=1000),
        "v2": load_ql(res_dir=source_dic[args.dataset]['dense'], top=100),
    }
    query, ans = get_qa(source_dic[args.dataset]['qa'])
    dall = get_dall(ql["v2"] + ql["bm25"], 100)
    dall = dall | drand
    doc = read_doc(doc_dir="source/para.title.txt", d_all=dall)
    title = read_doc(doc_dir="source/para.title.txt", d_all=dall)
    f = open(source_dic[args.dataset]['outfile'], 'w', encoding='utf-8')
    k = 0
    add_dic = {}
    for qid in tqdm(range(len(query))):
        q, a, cands = query[qid], ans[qid], ql["v2"][qid]
        positive_ctxs = []
        rand_negative_ctxs = []
        hard_negative_ctxs = []
        less_hard_negative_ctxs = []
        v2_ctxs = []
        bm25_ctxs = []
        neg_cands = []
        for did in cands:
            d = doc[did]
            t = title[did]
            if len(v2_ctxs) < 20:
                txt = gettxt(t, d)
                v2_ctxs.append(txt)
            if not has_answer(a, d):
                if len(hard_negative_ctxs) < 10:
                    txt = gettxt(t, d)
                    hard_negative_ctxs.append(txt)
                else:
                    neg_cands.append(did)
            else:
                if len(positive_ctxs) < 10:
                    txt = gettxt(t, d)
                    positive_ctxs.append(txt)
        set_less_hard_negative_ctxs = set()
        while len(less_hard_negative_ctxs) < min(10, len(neg_cands)):
            x = randint(0, len(neg_cands) - 1)
            x = neg_cands[x]
            d = doc[x]
            if x in set_less_hard_negative_ctxs:
                continue
            set_less_hard_negative_ctxs.add(x)
            t = title[x]
            if not has_answer(a, d):
                txt = gettxt(t, d)
                less_hard_negative_ctxs.append(txt)
        for did in ql['bm25'][qid][: 10]:
            d = doc[did]
            t = title[did]
            txt = gettxt(t, d)
            bm25_ctxs.append(txt)
        
        while len(rand_negative_ctxs) < 10:
            x = dr[k]
            k += 1
            t = title[x]
            d = doc[x]
            if not has_answer(a, d):
                txt = gettxt(t, d)
                rand_negative_ctxs.append(txt)
            else:
                if x not in cands:
                    if len(positive_ctxs) < 10:
                        txt = gettxt(t, d)
                        positive_ctxs.append(txt)
                        txt = gettxt(t, d)
                        positive_ctxs.append(txt)
                        if qid not in add_dic.keys():
                            add_dic[qid] = []
                        add_dic[qid].append(len(positive_ctxs) - 1)
        json.dump({'id': qid,
                  "question": q,
                  "reference": a,
                  'task': args.dataset.upper(),
                  'gold_ctxs': positive_ctxs,
                  'rand_ctxs': rand_negative_ctxs,
                  'strong_ctxs': hard_negative_ctxs,
                  'weak_ctxs': less_hard_negative_ctxs,
                  'dense_ctxs': v2_ctxs,
                  'sparse_ctxs': bm25_ctxs,
                  }, f, ensure_ascii=False)
        f.write('\n')
    f.close()
    json.dump(add_dic, open(source_dic[args.dataset]['outfile'] + 'add_dict.json', 'w', encoding='utf-8'), ensure_ascii=False)


if __name__ == "__main__":
    main()
