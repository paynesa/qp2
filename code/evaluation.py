import sys, os, argparse
import numpy as np 
LEMMA = 0
FEATS = 2
INFL = 1
FREQ = 3

def read_train(train_path, include_ftune=True):
   def get_seen(path, index):
      return set([line.split("\t")[index].strip() for line in open(path, "r").readlines()])
   seen_lemmas = get_seen(f"{train_path}.trn", 0)
   seen_fts = get_seen(f"{train_path}.trn", 2)
   if include_ftune: 
      seen_lemmas = seen_lemmas.union(get_seen(f"{train_path}.ftune", 0))
      seen_fts = seen_fts.union(get_seen(f"{train_path}.ftune", 2))
   return seen_lemmas, seen_fts


def read_res(res_path):
   return 


def evaluate(seen_lemmas, seen_fts, res_lines, ignore_fts_novel=True):
   seen_total = 0
   seen_correct = 0
   unseen_total = 0
   unseen_correct = 0

   for lemma, feats, correct in res_lines:
      if ignore_fts_novel and feats not in seen_fts:
         continue
      if lemma in seen_lemmas:
         seen_total += 1
         if correct:
            seen_correct += 1
      else: 
         unseen_total += 1
         if correct:
            unseen_correct += 1

   return seen_correct/seen_total*100, unseen_correct/unseen_total*100




def main(train_path, res_path):
   for family in [f for f in os.listdir(res_path) if "." not in f]:
      print(f"{family}\tLANG\tSEEN\tUNSEEN\tDIFFERENCE")
      for lang in sorted(set([l.strip().split(".")[0] for l in os.listdir(f"{res_path}/{family}")])):
         test_lines = [tuple(line.strip().split("\t")) for line in open(f"{train_path}/{family}/{lang}.tst").readlines()]
         decode_lines = [int(line.strip().split("\t")[-1]) for line in open(f"{res_path}/{family}/{lang}..decode.tsv").readlines()[1:]]
         res_lines = [(x[0], x[2], y == 0) for x, y in zip(test_lines, decode_lines)]
         seen_lemmas, seen_fts = read_train(f"{train_path}/{family}/{lang}")
         seen_pct, unseen_pct = evaluate(seen_lemmas, seen_fts, res_lines)
         print(f"\t\t{lang}:\t{seen_pct :.3f}\t{unseen_pct :.3f}\t{seen_pct - unseen_pct :.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate train/dev/test splits with controlled overlap")
    # Need separate arguments for the train & gold data because they're often in different folders 
    parser.add_argument("train_path", help = "Path to the directory containing the train/dev/test splits")
    parser.add_argument("res_path", help="Path to the directory containing the splits")
    args = parser.parse_args()
    main(args.train_path, args.res_path)