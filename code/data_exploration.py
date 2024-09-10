import numpy as np 
 
def parse_files(path, element=-1):
    """Helper function to get the train, dev, and test elements for a given path"""
    def get_element(full_path, element):
        return [line.strip().split("\t")[element] for line in open(full_path).readlines()]
    return get_element(f"{path}.trn", element), get_element(f"{path}.dev", element), get_element(f"{path}.tst", element)

def feature_overlap(path):
    """Calculate the percent of featuresets that were seen during training"""
    train_feats, dev_feats, test_feats = parse_files(path, -1)
    seen_feats = set(train_feats + dev_feats)
    proportion_seen = len([f for f in test_feats if f in seen_feats])/len(test_feats)
    return proportion_seen 

def unique_lemmas(path):
    """Return the number of unique lemmas in train + dev"""
    train_lemmas, dev_lemmas, test_lemmas = parse_files(path, 0)
    return len(set(train_lemmas + dev_lemmas))

def unique_featuresets(path):
    """Return the number of unique features in train + dev"""
    train_feats, dev_feats, test_feats = parse_files(path, -1)
    return len(set(train_feats + dev_feats))

def unique_features(path):
    """Return the number of unique features in train + dev"""
    train_feats, dev_feats, test_feats = parse_files(path, -1)
    unique_feats = set()
    for feat in train_feats + dev_feats:
        for f in feat.strip().split(";"):
            unique_feats.add(f)
    return len(unique_feats)

def train_size(path):
    """Calculate the training size for a given language"""
    train_feats, dev_feats, test_feats = parse_files(path, -1)
    assert(len(train_feats) != 0 and len(dev_feats) != 0 and len(test_feats) != 0)
    return len(train_feats + dev_feats)

def investigate_feature_overlap(path, verbose = True):
    """Helper function to investigate the feature overlap"""
    
    def get_pos_lemma_feats(feats, lemmas):
        """Inner helper function to extract a dictionary mapping POS to lemmas to features"""
        POS_lemmas_feats = {}
        for feat, lemma in zip(feats, lemmas):
            POS = feat.strip().split(";")[0]
            if POS not in POS_lemmas_feats:
                POS_lemmas_feats[POS] = {}
            if lemma not in POS_lemmas_feats[POS]:
                POS_lemmas_feats[POS][lemma] = set()
            POS_lemmas_feats[POS][lemma].add(feat)
        return POS_lemmas_feats
            
        
    def get_paradigm_sizes(lemmas_to_feats):
        """Inner helper function to get the mean and standard deviation of the paradigm sizes for a POS"""
        paradigm_sizes = np.asarray([len(x) for x in lemmas_to_feats.values()])
        mean = np.average(paradigm_sizes)
        stdev = np.std(paradigm_sizes)
        return mean, stdev
        
    # Extract the lemmas and features for the train, dev, and test sets 
    train_feats, dev_feats, test_feats = parse_files(path, -1)
    train_lemmas, dev_lemmas, test_lemmas = parse_files(path, 0)
    
    # Get dictionaries mapping POS to lemmas to features for train & test 
    POS_train = get_pos_lemma_feats(train_feats + dev_feats, train_lemmas + dev_lemmas)
    POS_test = get_pos_lemma_feats(test_feats, test_lemmas)
    
    # Collect all lemmas that appear with at least one unseen featureset in test
    # We use lemmas because a single lemma often appears with a great number of unseen features 
    seen_feats = set(train_feats + dev_feats)
    relevant_lemmas = set()
    for feat, lemma in zip(test_feats, test_lemmas):
        if feat not in seen_feats: 
            relevant_lemmas.add(lemma)
            
    # Iterate through each POS that was seen in training 
    results = {}
    for POS in POS_train:
        results[POS] = {}
        # Get the mean paradigm size and variation in the paradigm size for the training data 
        mean_size_train, stdev_train = get_paradigm_sizes(POS_train[POS])
        if verbose:
            print(f"POS: {POS}\n\t mean train paradigm size:\t {mean_size_train :.3f}, (stdev: {stdev_train :.3f}, n = {len(POS_train[POS])}, max = {max([len(x) for x in POS_train[POS].values()])})")
        results[POS]["train"] = (mean_size_train, stdev_train, max([len(x) for x in POS_train[POS].values()]))

        # If that POS was attested in test, then also calculate the overall test size 
        if POS in POS_test:
            mean_size_test, stdev_test = get_paradigm_sizes(POS_test[POS])
            if verbose:
                print(f"\t mean test paradigm size:\t {mean_size_test :.3f}, (stdev: {stdev_test :.3f}, n = {len(POS_test[POS])})")
            results[POS]["test"] = (mean_size_test, stdev_test, max([len(x) for x in POS_test[POS].values()]))

            # If there are any unattested features for this POS, get them and find the relevant sizes 
            relevant_pairs = {k: v for k, v in POS_test[POS].items() if k in relevant_lemmas}
            if relevant_pairs:
                mean_size_relevant, stdev_relevant = get_paradigm_sizes(relevant_pairs)
                if verbose:
                    print(f"\t mean problematic lemma size:\t {mean_size_relevant :.3f}, (stdev: {stdev_relevant :.3f}, n = {len(relevant_pairs)})")
                results[POS]["problematic"] = (mean_size_relevant, stdev_relevant)
            elif verbose:
                print("\t no problematic lemmas for this POS")
        elif verbose:
            print("\t POS not attested in test")
        if verbose:
            print("\n")    

    return results  

def overlap_less(second, ovlp):
    """Filter two lists to just the positions where overlap is less than 100"""
    ovlp_new = []
    second_new = []
    for o, s in zip(ovlp, second):
        if o < 1:
            ovlp_new.append(o)
            second_new.append(s) 
    return second_new, ovlp_new

