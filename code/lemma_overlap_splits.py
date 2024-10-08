import sys, os, argparse, random
LEMMA = 0
FEATS = 2
INFL = 1
FREQ = 3



def compute_overlap(train, test, overlap_item, printoverlap=False):
   """Compute the overlap between the train and test data"""
   trainitems = [triple[overlap_item] for triple in train]
   testitems = [triple[overlap_item] for triple in test]
   numtestoverlap = len([i for i in testitems if i in trainitems])
   numtrainoverlap = len([i for i in trainitems if i in testitems])
   
   if printoverlap:
      print(f"\t\tOverlap in train: {100*numtrainoverlap/len(trainitems)}")
      print(f"\t\tOverlap in test: {100*numtestoverlap/len(testitems)}")

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



def read_corpus(train_path, gold_path, overlap_item):
   """Read in the data for a single language and create a mapping from the relevant overlap items to the triples containing that item"""
   def parse_unimorph(path):
      return [tuple(line.strip().split("\t")) for line in open(path, "r").readlines()]
   # Read in the train & dev data & the gold test data, which is in a different folder 
   train_lines = parse_unimorph(f"{train_path}.trn")
   dev_lines = parse_unimorph(f"{train_path}.dev")
   test_lines = parse_unimorph(f"{gold_path}.tst")
   # Get the set of all lines 
   all_lines = set(train_lines + dev_lines + test_lines)
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

   print(f"\t\tNum sampled: {len(sampled)}; Num remaining: {len(remaining)}")
   return sampled, remaining




def controlled_overlap_sample(line_dict, triples, trainsize, testsize, ftuneprop, overlap_item, overlap_ratio):
   """This function executes overlap-aware sampling"""
   all_items = sorted(line_dict.keys())
   random.shuffle(all_items)
   partition = int(overlap_ratio * len(all_items))
   origpartition = partition

   # Iteratively increase the partition until we can sample enough 
   trainsample = {}
   print("\t\tSampling train...")
   while len(trainsample) < trainsize:
      # Log if we needed to increase the partition for train
      if partition == origpartition + 1:
         print(f"\t\tMust oversample large train. Gap: {trainsize -len(trainsample)}")
      # Get the items that are overlappable 
      overlappable = set(all_items[:partition])
      # Get the items that contain the overlappable items
      overlaptriples = [triple for triple in triples if triple[overlap_item] in overlappable]
      # Shuffle the overlappable triples and sample up to the training size
      random.shuffle(overlaptriples)
      trainsample = overlaptriples[:trainsize]
      # Get all the triples not in the training data
      remaining = sorted(set(triples).difference(trainsample))
      partition += 1

   # Get all the overlappable items that are present in the training data 
   items_in_train = set(triple[overlap_item] for triple in trainsample)

   # Split the training data into train & finetune
   random.shuffle(trainsample)
   cutoff = int(ftuneprop * trainsize)
   train = trainsample[:cutoff]
   ftune = trainsample[cutoff:]

   print("\t\tTrain sampled. Sampling test...")


   test, remaining = subsample(line_dict, remaining, items_in_train, testsize, overlap_ratio, overlap_item)
   print("\t\tTest sampled.")

   return train, ftune, test 



def main(train_path, gold_path, trainsize, testsize, overlap_item, overlap_ratio, ftuneprop, seed):
   """The main function to execute the splitting"""
   print(f"Training size: {trainsize} (ftune subset: {ftuneprop*trainsize}), test size: {testsize}. Seed = {seed}")
   random.seed(seed)
   for family in [f for f in os.listdir(train_path) if "." not in f]:
      print(f"Splitting {family} family...")
      for lang in set([l.strip().split(".")[0] for l in os.listdir(f"{train_path}/{family}")]):
         lines, line_dict = read_corpus(f"{train_path}/{family}/{lang}", f"{gold_path}/{lang}", overlap_item)
         # Only attempt sampling if it's at least big enough 
         if len(lines) >= trainsize + testsize:
            print(f"\tSplitting {lang} ({len(lines)} triples)...")
            train, ftune, test = controlled_overlap_sample(line_dict, lines, trainsize, testsize, ftuneprop, overlap_item, overlap_ratio)
            validate(train, ftune, test, overlap_item, printoverlap=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make train / dev / test split with controlled overlap")
    # Need separate arguments for the train & gold data because they're often in different folders 
    parser.add_argument("train_data", help = "Path to the directory containing the data to be split")
    parser.add_argument("gold_data", help="Path to the directory containing the gold test data")
    parser.add_argument("train_size", help="The number of triples to sample for train + ftune", type = int)
    parser.add_argument("test_size", help = "The number of triples to sample for test", type = int)

    # Optional arguments that we have reasonable defaults for 
    parser.add_argument("--overlap_ratio", help = "The maximum ratio of overlap", type = float, default = 0.5)
    parser.add_argument("--overlap_item",  help = "The item whose overlap should be controlled", default="LEMMA")
    parser.add_argument("--ftune_prop", help = "The proportion of items in train to be subsampled for ftune", type=float, default = 0.2)
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
      args.seed
      )


