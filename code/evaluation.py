import sys, os, argparse
import numpy as np 
import pandas as pd 

LEMMA = 0
FEATS = 2
INFL = 1
FREQ = 3

def read_train(train_path, include_ftune=True):
   """Helper function to read in the training data to get a set of the attested & unattested lemmas & featuresets"""
   def get_seen(path, index):
      return set([line.split("\t")[index].strip() for line in open(path, "r").readlines()])
   # Get the lemmas and feature sets in train
   seen_lemmas = get_seen(f"{train_path}.trn", 0)
   seen_fts = get_seen(f"{train_path}.trn", 2)
   # If presence in finetune also constitutes presence in train, then include the lemmas and features seen there 
   if include_ftune: 
      seen_lemmas = seen_lemmas.union(get_seen(f"{train_path}.ftune", 0))
      seen_fts = seen_fts.union(get_seen(f"{train_path}.ftune", 2))
   return seen_lemmas, seen_fts



def evaluate(seen_lemmas, seen_fts, res_lines, ignore_fts_novel=True):
   """The main evaluation function"""
   seen_total = 0
   seen_correct = 0
   unseen_total = 0
   unseen_correct = 0
   for lemma, feats, correct in res_lines:
      # If we're ignoring elements with novel features in our evaluation, then drop them here 
      if ignore_fts_novel and feats not in seen_fts:
         continue
      # If the triple's lemma has been seen in train, tally accordingly 
      if lemma in seen_lemmas:
         seen_total += 1
         if correct:
            seen_correct += 1
      # If the triple's lemma has not been seen in train, tally accordingly 
      else: 
         unseen_total += 1
         if correct:
            unseen_correct += 1
   # Calculate & return percents 
   return seen_correct/seen_total*100, unseen_correct/unseen_total*100




def main(train_path, res_path, out_path):
   """The main function which executes the evaluation loop """
   train_path = train_path[:-1] if train_path[-1] == "/" else train_path
   res_path = res_path[:-1] if res_path[-1] == "/" else res_path
   evaluation_dict = {}
   for family in [f for f in os.listdir(res_path) if "." not in f]:
      print(f"Processing {family}...")
      for lang in sorted(set([l.strip().split(".")[0] for l in os.listdir(f"{res_path}/{family}")])):
         # Read in the lines from the testing file input to the model
         test_lines = [tuple(line.strip().split("\t")) for line in open(f"{train_path}/{family}/{lang}.tst").readlines()]
         # Read in the lines from the decoded file 
         decode_lines = [int(line.strip().split("\t")[-1]) for line in open(f"{res_path}/{family}/{lang}/{lang}..decode.tsv").readlines()[1:]]
         # Combine to (lemma, feature, correct) triples
         res_lines = [(x[0], x[2], y == 0) for x, y in zip(test_lines, decode_lines)]
         # Get the set of seen lemmas and features in train + dev to conduct the evaluation 
         seen_lemmas, seen_fts = read_train(f"{train_path}/{family}/{lang}", include_ftune=True)
         seen_pct_t, unseen_pct_t = evaluate(seen_lemmas, seen_fts, res_lines, ignore_fts_novel=True)
         seen_pct_f, unseen_pct_f = evaluate(seen_lemmas, seen_fts, res_lines, ignore_fts_novel=False)
         # Get the set of lemmas and features in train only to conduct the evaluation
         train_lemmas, train_fts = read_train(f"{train_path}/{family}/{lang}", include_ftune=False)
         train_pct_t, untrain_pct_t = evaluate(train_lemmas, train_fts, res_lines, ignore_fts_novel=True)
         train_pct_f, untrain_pct_f = evaluate(train_lemmas, train_fts, res_lines, ignore_fts_novel=False)
         
         evaluation_dict[lang] = {
            "family": family,
            "seen_pct_t": seen_pct_t,
            "unseen_pct_t": unseen_pct_t,
            "seen_pct_f": seen_pct_f,
            "unseen_pct_f": unseen_pct_f,
            "train_pct_t": train_pct_t,
            "untrain_pct_t": untrain_pct_t,
            "train_pct_f": train_pct_f,
            "untrain_pct_f": untrain_pct_f,
            "seen_lemmas": len(set(seen_lemmas)),
            "train_lemmas": len(set(train_lemmas))
         }
   print("Writing output...")
   evaluation_df = pd.DataFrame.from_dict(evaluation_dict).transpose()
   evaluation_df.to_csv(out_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate train/dev/test splits with controlled overlap")
    parser.add_argument("train_path", help = "Path to the directory containing the train/dev/test splits")
    parser.add_argument("res_path", help="Path to the directory containing the splits")
    parser.add_argument("out_path", help="Path to write the results to")
    args = parser.parse_args()
    main(args.train_path, args.res_path, args.out_path)