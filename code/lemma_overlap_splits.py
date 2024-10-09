import sys, os, argparse, random
import numpy as np 
LEMMA = 0
FEATS = 2
INFL = 1
FREQ = 3



def write_splits(outdir, family, lang, seed, train, ftune, test):
   """This function writes out the files"""
   if not os.path.exists(f"{outdir}/{seed}/{family}"):
      os.makedirs(f"{outdir}/{seed}/{family}")
   with open(f"{outdir}/{seed}/{family}/{lang}.trn", "a") as trn:
      for t1, t2, t3 in train:
         trn.write(f"{t1}\t{t2}\t{t3}\n")
   trn.close()
   with open(f"{outdir}/{seed}/{family}/{lang}.ftune", "a") as ft:
      for t1, t2, t3 in ftune:
         ft.write(f"{t1}\t{t2}\t{t3}\n")
   ft.close()
   with open(f"{outdir}/{seed}/{family}/{lang}.tst", "a") as tst:
      for t1, t2, t3 in test:
         tst.write(f"{t1}\t{t2}\t{t3}\n")
   ft.close()


def compute_overlap(train, test, overlap_item, printoverlap=False):
   """Compute the overlap between the train and test data"""
   trainitems = [triple[overlap_item] for triple in train]
   testitems = [triple[overlap_item] for triple in test]
   numtestoverlap = len([i for i in testitems if i in trainitems])
   numtrainoverlap = len([i for i in trainitems if i in testitems])
   
   if printoverlap:
      print(f"\t\t{'Lemma' if overlap_item == LEMMA else 'Feature'} overlap in train: {100*numtrainoverlap/len(trainitems)}")
      print(f"\t\t{'Lemma' if overlap_item == LEMMA else 'Feature'} overlap in test: {100*numtestoverlap/len(testitems)}")

   return 100*numtrainoverlap/len(trainitems), 100*numtestoverlap/len(testitems)


def validate(train, ftune, test, overlap_item, printoverlap=False):
   """Validate our train / ftune / test splits"""
   train_all = set(train).union(set(ftune))
   testset = set(test)
   assert(len(train_all.intersection(testset)) == 0)
   if printoverlap:
      print(f"\t\tTriple overlap between train and test: {len(train_all.intersection(testset))}")
   train_overlap, test_overlap = compute_overlap(train_all, testset, overlap_item=overlap_item, printoverlap=printoverlap)
   assert(test_overlap <= 50)
   other = [x for x in (LEMMA, FEATS)if x != overlap_item][0]
   train_overlap2, test_overlap2 = compute_overlap(train_all, testset, overlap_item = other, printoverlap = False)
   print(f"\t\t{'Lemma' if other == LEMMA else 'Feature'} overlap in test: {test_overlap2}")
   return test_overlap


def read_corpus(train_path, gold_path, overlap_item):
   """Read in the data for a single language and create a mapping from the relevant overlap items to the triples containing that item"""
   def parse_unimorph(path):
      return [tuple(line.strip().split("\t")) for line in open(path, "r").readlines()]
   # Read in the train & dev data & the gold test data, which is in a different folder 
   train_lines = parse_unimorph(f"{train_path}.trn")
   dev_lines = parse_unimorph(f"{train_path}.dev")
   test_lines = parse_unimorph(f"{gold_path}.tst")
   # Get the set of all lines 
   all_lines = sorted(set(train_lines + dev_lines + test_lines))
   # Get the mapping from the relevant overlap item to the triples containing it
   line_dict = {}
   for triple in all_lines:
      if triple[overlap_item] not in line_dict:
         line_dict[triple[overlap_item]] = set()
      line_dict[triple[overlap_item]].add(triple)
   return all_lines, line_dict


def subsample(line_dict, triples, overlappable, numsample, overlap_ratio, overlap_item):
   """Helper function to sample our dev and test once we've sampled train"""

   # Get the triples that contain or don't contain the given overlap item & shuffle them
   overlaptriples = [triple for triple in triples if triple[overlap_item] in overlappable]
   nonoverlaptriples = [triple for triple in triples if triple[overlap_item] not in overlappable]
   random.shuffle(overlaptriples)
   random.shuffle(nonoverlaptriples)

   # Get the number of overlappable and non-overlappable triples we want to sample 
   num_overlappable = int(numsample * overlap_ratio)
   num_nonoverlappable = numsample - num_overlappable

   # Sample that many overlap triples & non-overlap triples 
   sampled = overlaptriples[:num_overlappable] + nonoverlaptriples[:num_nonoverlappable]
   remaining = overlaptriples[num_overlappable:] + nonoverlaptriples[num_nonoverlappable:]


   # If enough triples aren't sampled, then increase the sample along the larger subset
   if len(sampled) < numsample: 
      gap = numsample - len(sampled)
      print(f"\t\tMust oversample test. gap: {gap}")
      # If there aren't enough non-overlappable triples, sample more of the overlappable ones 
      if len(nonoverlaptriples) < num_nonoverlappable:
         sampled = overlaptriples[:num_overlappable + gap] + nonoverlaptriples[:num_nonoverlappable]
         remaining = overlaptriples[num_overlappable + gap:] + nonoverlaptriples[num_nonoverlappable:]
      # If there aren't enough overlappable triples, sample more of the non-overlappable ones 
      elif len(overlaptriples) < num_overlappable:
         sampled = nonoverlaptriples[:num_overlappable + gap] + overlaptriples[:num_nonoverlappable]
         remaining = nonoverlaptriples[num_overlappable + gap:] + overlaptriples[num_nonoverlappable:]
   return sampled, remaining


def controlled_overlap_sample(line_dict, triples, trainsize, testsize, ftuneprop, overlap_item, overlap_ratio):
   """This function executes overlap-aware sampling"""
   all_items = sorted(line_dict.keys())
   random.shuffle(triples)
   random.shuffle(all_items)
   partition = int(overlap_ratio * len(all_items))
   origpartition = partition

   # Iteratively increase the partition until we can sample enough 
   total_overlappable = 0
   overlappable = {}
   print("\t\tSampling train...")

   # Iterate until we have sufficiently many triples with the overlap features
   while total_overlappable < trainsize + overlap_ratio * testsize and len(overlappable) < len(all_items):
      # Log if we needed to increase the partition for train
      if partition == origpartition + 1:
         print(f"\t\tMust oversample large train. Gap: {trainsize + overlap_ratio * testsize -len(overlaptriples)}")
      # Get the items that are overlappable 
      overlappable = set(all_items[:partition])
      # Get the items that contain the overlappable items
      overlaptriples = [triple for triple in triples if triple[overlap_item] in overlappable]
      # Sample the training data as a subset of the triples with overlap 
      random.shuffle(overlaptriples)
      trainsample = overlaptriples[:trainsize]
      # Now find how many triples have the relevant overlap item 
      items_in_train = set(triple[overlap_item] for triple in trainsample)
      total_overlappable = len([t for t in triples if t[overlap_item] in items_in_train])
      partition += 1

   # Get all the remaining items 
   remaining = sorted(set(triples).difference(trainsample))

   # Split the training data into train & finetune
   random.shuffle(trainsample)
   cutoff = int(ftuneprop * trainsize)
   train = trainsample[cutoff:]
   ftune = trainsample[:cutoff]

   print("\t\tTrain sampled. Sampling test...")


   test, remaining = subsample(line_dict, remaining, items_in_train, testsize, overlap_ratio, overlap_item)
   print("\t\tTest sampled.")

   return train, ftune, test 



def main(train_path, gold_path, trainsize, testsize, overlap_item, overlap_ratio, ftuneprop, seed, outdir):
   """The main function to execute the splitting"""
   print(f"Training size: {trainsize} (ftune subset: {ftuneprop*trainsize}), test size: {testsize}. Seed = {seed}")
   # Set the random seed for replicability 
   random.seed(seed)
   # Create the output directory if it doesn't already exist 
   if not os.path.exists(f"{outdir}/{seed}"):
      os.makedirs(f"{outdir}/{seed}")
   # Store languages which don't achieve the overlap ratio
   languages_with_lower_overlap = []
   # Consider languages family-by-family 
   for family in [f for f in os.listdir(train_path) if "." not in f]:
      print(f"Splitting {family} family...")
      for lang in set([l.strip().split(".")[0] for l in os.listdir(f"{train_path}/{family}")]):
         # Read in the corpus
         lines, line_dict = read_corpus(f"{train_path}/{family}/{lang}", f"{gold_path}/{lang}", overlap_item)
         # Only attempt sampling if it's at least big enough 
         if len(lines) >= trainsize + testsize:
            print(f"\tSplitting {lang} ({len(lines)} triples)...")
            sizes = np.asarray([len(v) for v in line_dict.values()])
            print(f"\t\tMean size: {np.mean(sizes) :.3f} (stdev: {np.std(sizes) :.3f}, n: {len(sizes)})")
            train, ftune, test = controlled_overlap_sample(line_dict, lines, trainsize, testsize, ftuneprop, overlap_item, overlap_ratio)
            test_overlap = validate(train, ftune, test, overlap_item, printoverlap=True)
            # If we achieve the desired overlap, write out the result. Otherwise, warn the user
            if test_overlap < overlap_ratio:
               languages_with_lower_overlap.append((lang, test_overlap))
            else:
               print(f"\t\tWriting splits to {outdir}")
               write_splits(outdir, family.lower(), lang, seed, train, ftune, test)
   print(f"Finished splitting. The following languages didn't reach {overlap_ratio} overlap:")
   for language, overlap in languages_with_lower_overlap:
      print(f"\t{language} ({overlap} overlap)")
   print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make train / dev / test split with controlled overlap")
    # Need separate arguments for the train & gold data because they're often in different folders 
    parser.add_argument("train_data", help = "Path to the directory containing the data to be split")
    parser.add_argument("gold_data", help="Path to the directory containing the gold test data")
    parser.add_argument("train_size", help="The number of triples to sample for train + ftune", type = int)
    parser.add_argument("test_size", help = "The number of triples to sample for test", type = int)
    parser.add_argument("outdir", help = "The directory to which the splits should be written")

    # Optional arguments that we have reasonable defaults for 
    parser.add_argument("--overlap_ratio", help = "The maximum ratio of overlap", type = float, default = 0.5)
    parser.add_argument("--overlap_item",  help = "The item whose overlap should be controlled", default="LEMMA")
    parser.add_argument("--ftune_prop", help = "The proportion of items in train to be subsampled for ftune", type=float, default = 0.125)
    parser.add_argument("--seed", help = "The random seed", type = int, default = 1)
    args = parser.parse_args()

    # Parse the arguments and call the main function
    if args.overlap_item.strip().upper() == "LEMMA":
      print("Splitting to control lemma overlap")
      overlap_item = LEMMA
    elif args.overlap_item.strip().upper() == "FEATS":
      print("Splitting to control feature overlap")
      overlap_item = FEATS
    else:
      raise Exception("Overlap must be either lemma or features")
    main(args.train_data, 
      args.gold_data, 
      args.train_size, 
      args.test_size, 
      overlap_item, 
      args.overlap_ratio, 
      args.ftune_prop,
      args.seed,
      args.outdir
      )


