import sys, os, argparse, random
from math import ceil
from numpy.random import choice, seed
from collections import defaultdict

LEMMA = 0
FEATS = 2
INFL = 1
FREQ = 3

def set_seed(randomseed):
    random.seed(randomseed) #random seed
    seed(sum([ord(c) for c in randomseed])) #numpy random seed


def readcorpus(infname):
    feats_to_triples = defaultdict(lambda: set)
    triples = set([])
    triples_to_freqs = {}
    seenlemmafeats = set([]) # Remove duplicate lemma, feature pairs that can exist in UniMorph. Very confusing.
    with open(infname, "r") as fin:
        for line in fin:
            if not line.strip():
                continue
            lemma = ""
            infl = ""
            feats = ""
            freq = 0
            try:
                lemma, infl, feats, freq = line.strip().split("\t")
            except ValueError:
                lemma, infl, feats = line.strip().split("\t")
            if (lemma, feats) in seenlemmafeats:
                continue
            seenlemmafeats.add((lemma, feats))
            freq = float(freq)
            triples_to_freqs[(lemma, infl, feats)] = freq
            feats_to_triples[feats] = (lemma, infl, feats)
            triples.add((lemma, infl, feats))
    indices_to_triples = {i:triple for i, triple in enumerate(triples_to_freqs)}
    return sorted(triples), triples_to_freqs, feats_to_triples, indices_to_triples


def writesample(sample, showinfl, outdir, fname):
    outfname = os.path.join(outdir, fname)
    cutoff = 3
    if not showinfl:
        cutoff = 2
    with open(outfname, "w") as fout:
        for s in sample:
            if showinfl:
                fout.write("%s\n" % "\t".join(s))
            else:
                fout.write("%s\t%s\n" % (s[LEMMA], s[FEATS]))


def writesamples(outdir, language, seed, ltrain, lftune, dev, test, strain, sftune):
    writesample(strain, True, outdir,  "%s_%s_small.train" % (language, seed))
    writesample(ltrain, True, outdir, "%s_%s_large.train" % (language, seed))
    writesample(sftune, True, outdir,  "%s_%s_small.ftune" % (language, seed))
    writesample(lftune, True, outdir, "%s_%s_large.ftune" % (language, seed))
    writesample(test, False, outdir, "%s_%s.test" % (language, seed))
    writesample(test, True, outdir, "%s_%s.gold" % (language, seed))
    writesample(dev, True, outdir,  "%s_%s.dev" % (language, seed))


def compute_overlap(train, test, i=FEATS, printoverlap=False):
    trainitems = [triple[i] for triple in train]
    testitems = [triple[i] for triple in test]
    numtestoverlap = 0
    numtrainoverlap = 0
    for item in testitems:
        if item in trainitems:
            numtestoverlap += 1
    for item in trainitems:
        if item in testitems:
            numtrainoverlap += 1
    if printoverlap:
        print("%overlap in train", 100*numtrainoverlap/len(trainitems))
        print("%overlap in test", 100*numtestoverlap/len(testitems))
    return 100*numtrainoverlap/len(trainitems), 100*numtestoverlap/len(testitems)


def compute_overlaps(ltrain, lftune, strain, sftune, dev, test):
    ltrain_all = ltrain + lftune
    strain_all = strain + sftune

    #print("Achieved large-test feature overlap:")
    test_in_ltrainf, ltrain_in_testf = compute_overlap(ltrain_all, test, i=FEATS, printoverlap=False)
    #print("Achieved large-test lemma overlap:")
    test_in_ltrainl, ltrain_in_testl = compute_overlap(ltrain_all, test, i=LEMMA, printoverlap=False)
    #print("Achieved large-dev feature overlap:")
    dev_in_ltrainf, ltrain_in_devf = compute_overlap(ltrain_all, dev, i=FEATS, printoverlap=False)
    #print("Achieved large-dev lemma overlap:")
    dev_in_ltrainl, ltrain_in_devl = compute_overlap(ltrain_all, dev, i=LEMMA, printoverlap=False)

    #print("Achieved small-test feature overlap:")
    test_in_strainf, strain_in_testf = compute_overlap(strain_all, test, i=FEATS, printoverlap=False)
    #print("Achieved small-test lemma overlap:")
    test_in_strainl, strain_in_testl = compute_overlap(strain_all, test, i=LEMMA, printoverlap=False)
    #print("Achieved small-dev feature overlap:")
    dev_in_strainf, strain_in_devf = compute_overlap(strain_all, dev, i=FEATS, printoverlap=False)
    #print("Achieved small-dev lemma overlap:")
    dev_in_strainl, strain_in_devl = compute_overlap(strain_all, dev, i=LEMMA, printoverlap=False)

    #print("Achieved dev-test feature overlap:")
    test_in_devf, dev_in_testf = compute_overlap(dev, test, i=FEATS, printoverlap=False)
    #print("Achieved dev-test lemma overlap:")
    test_in_devl, dev_in_testl = compute_overlap(dev, test, i=LEMMA, printoverlap=False)

    return (test_in_ltrainf, ltrain_in_testf, test_in_ltrainl, ltrain_in_testl,
            dev_in_ltrainf, ltrain_in_devf, dev_in_ltrainl, ltrain_in_devl, 
            test_in_strainf, strain_in_testf, test_in_strainl, strain_in_testl,
            dev_in_strainf, strain_in_devf, dev_in_strainl, strain_in_devl, 
            test_in_devf, dev_in_testf, test_in_devl, dev_in_testl)


def subsample(feats_to_triples, triples, overlappablefeats, numsample, overlapratio):
    overlaptriples = [triple for triple in triples if triple[FEATS] in overlappablefeats]
    nonoverlaptriples = [triple for triple in triples if triple[FEATS] not in overlappablefeats]
    num_overlappable = int(numsample * overlapratio)
    num_nonoverlappable = numsample - num_overlappable
    random.shuffle(overlaptriples)
    random.shuffle(nonoverlaptriples)
#    print(num_overlappable,  num_nonoverlappable)
#    print(len(overlaptriples), len(nonoverlaptriples))
    sampled = overlaptriples[:num_overlappable] + nonoverlaptriples[:num_nonoverlappable]
    remaining = overlaptriples[num_overlappable:] + nonoverlaptriples[num_nonoverlappable:]
    if len(sampled) < numsample:
        gap = numsample-len(sampled)
        print("Must oversample. gap:", gap)
        if len(nonoverlaptriples) < num_nonoverlappable:
            sampled = overlaptriples[:num_overlappable + gap] + nonoverlaptriples[:num_nonoverlappable]
            remaining = overlaptriples[num_overlappable + gap:] + nonoverlaptriples[num_nonoverlappable:]
        elif len(overlaptriples) < num_overlappable:
            sampled = nonoverlaptriples[:num_overlappable + gap] + overlaptriples[:num_nonoverlappable]
            remaining = nonoverlaptriples[num_overlappable + gap:] + overlaptriples[num_nonoverlappable:]
        print("Num sampled: ", len(sampled), "Num remaining: ", len(remaining))
    return sampled, remaining


def validate(ltrain, lftune, dev, test, strain, sftune):
    ltrain_all = set(ltrain).union(set(lftune))
    strain_all = set(strain).union(set(sftune))
    devset = set(dev)
    testset = set(test)
    ltdevo = len(ltrain_all.intersection(devset))
    lttesto = len(ltrain_all.intersection(testset))
    stdevo = len(strain_all.intersection(devset))
    sttesto = len(strain_all.intersection(testset))
    devtesto = len(devset.intersection(testset))
#    print("Illicit large train triples in dev?", ltdevo)
#    print("Illicit large train triples in test?", lttesto)
#    print("Illicit small train triples in dev?", stdevo)
#    print("Illicit small train triples in test?", sttesto)
#    print("Illicit dev triples in test?", devtesto)
    assert(ltdevo == 0)
    assert(lttesto == 0)
    assert(stdevo == 0)
    assert(sttesto == 0)
    assert(devtesto == 0)
    return lttesto, ltdevo, sttesto, stdevo, devtesto


def naive_sample(triples_to_freqs, indices_to_triples, testsize, dsize, ltsize, lftsize, stsize, sftsize, weight):
    indices, triples = zip(*tuple(indices_to_triples.items()))
    if weight:
        counts = [triples_to_freqs[indices_to_triples[i]] for i in indices]
        sumcounts = sum(counts)
        probs = [count/sumcounts for count in counts]
        samplei = choice(indices, len(indices), replace=False, p=probs)
        sample = [indices_to_triples[i] for i in samplei]
    else:
        sample = list(triples)
        random.shuffle(sample)

    ltrain_all = sample[:ltsize+lftsize]
    #sample lftune uniformly with respect to ltrain
    random.shuffle(ltrain_all)
    ltrain = ltrain_all[:ltsize]
    lftune = ltrain_all[ltsize:]
    
    strain_all = ltrain[0:stsize+sftsize]
    #sample sftune uniformly with respect to strain
    random.shuffle(strain_all)
    strain = strain_all[:stsize]
    sftune = strain_all[stsize:]

    devtest = sample[len(ltrain_all):len(ltrain_all)+dsize+testsize]
    # Dev and test should be sampled uniformly with respect to one another
    random.shuffle(devtest)
    dev = devtest[:dsize]
    test = devtest[dsize:]

    return ltrain, lftune, dev, test, strain, sftune


def nofreq_feataware_sample(feats_to_triples, triples, testsize, dsize, ltsize, lftsize, stsize, sftsize, feat_overlap_ratio):

    featlist = sorted(feats_to_triples.keys())
    random.shuffle(featlist)
    partition = int(feat_overlap_ratio * len(featlist))
    origpartition = partition

    numlargetrain = ltsize + lftsize
    numsmalltrain = stsize + sftsize

    if numlargetrain != numsmalltrain:
        print("sampling large train...")
    else:
        print("sampling small train...")

    largetrainsample = {}
    while len(largetrainsample) < numlargetrain:
        if partition == origpartition + 1:
            print("Must oversample large train. gap:", numlargetrain-len(largetrainsample))
        overlappablefeats = set(featlist[:partition])
        overlaptriples = [triple for triple in triples if triple[FEATS] in overlappablefeats]
#        print(len(overlaptriples))
        random.shuffle(overlaptriples)
        largetrainsample = overlaptriples[:numlargetrain]
#        print(len(largetrainsample))
        remaining = sorted(set(triples).difference(largetrainsample))
        partition += 1
    feats_in_train = set([triple[FEATS] for triple in largetrainsample])

    random.shuffle(largetrainsample)
    ltrain = largetrainsample[:ltsize]
    lftune = largetrainsample[ltsize:]

    if numlargetrain != numsmalltrain:
        print("sampling small train...")
    smalltrainsample, _ = subsample(feats_to_triples, ltrain, feats_in_train, numsmalltrain, 1)
    random.shuffle(smalltrainsample)
    strain = smalltrainsample[:stsize]
    sftune = smalltrainsample[stsize:]

    print("sampling test...")
    test, remaining = subsample(feats_to_triples, remaining, feats_in_train, testsize, feat_overlap_ratio)

    print("sampling dev...")
    feats_in_test = set([triple[FEATS] for triple in test])
    dev, remaining = subsample(feats_to_triples, remaining, feats_in_test, dsize, feat_overlap_ratio)
    if numlargetrain != numsmalltrain:
        print("Achieved Sizes (lt, lft, st, sft, d, test):", len(ltrain), len(lftune), len(strain), len(sftune), len(dev), len(test))
    else:
        print("Achieved Sizes (st, sft, d, test):", len(strain), len(sftune), len(dev), len(test))

    return ltrain, lftune, dev, test, strain, sftune


def init_log(outdir):
    logfname = os.path.join(outdir, "log.tsv")
    return open(logfname, "a")


def log(logger, language, sampletype, seed, ltrain, lftune, dev, test, strain, sftune, overlaps, illicitoverlaps):
    logger.write("%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t" % (language, sampletype, seed, len(ltrain), len(lftune), len(strain), len(sftune), len(dev), len(test)))
    logger.write("\t".join([str(o) for o in overlaps]))
    logger.write("\t")
    logger.write("\t".join([str(o) for o in illicitoverlaps]))
    logger.write("\n")


def close_log(logger):
    logger.close()


def main(freqfname, rawfname, outdir, stsize, ltsize, sftsize, lftsize, dsize, testsize, foverlap, language, seed):
    logger = init_log(outdir)
    print("Input:", freqfname)
    print("Input:", rawfname)
    triples, triples_to_freqs, feats_to_triples, indices_to_triples = readcorpus(freqfname)
    _, triples_to_0_raw, _, indices_to_triples_raw = readcorpus(rawfname)
    print("Output:", outdir)
    print("Seed:", seed)
    print("contains %s items" % len(triples))
    print("contains %s unique feature sets" % len(feats_to_triples))
    print("small train:", stsize+sftsize, "large train:", ltsize+lftsize, "dev:", dsize, "test:", testsize)

    print("Naive Uniform Sampling...")
    set_seed(seed)
    ltrain_uni, lftune_uni, dev_uni, test_uni, strain_uni, sftune_uni = naive_sample(triples_to_0_raw, indices_to_triples_raw, testsize, dsize, ltsize, lftsize, stsize, sftsize, weight=False)
    overlaps = compute_overlaps(ltrain_uni, lftune_uni, strain_uni, sftune_uni, dev_uni, test_uni)
    illicitoverlaps = validate(ltrain_uni, lftune_uni, dev_uni, test_uni, strain_uni, sftune_uni)
    log(logger, language, "naive_uni", seed, ltrain_uni, lftune_uni, dev_uni, test_uni, strain_uni, sftune_uni, overlaps, illicitoverlaps)
    writesamples(os.path.join(outdir,"naive_uniform"), language, seed, ltrain_uni, lftune_uni, dev_uni, test_uni, strain_uni, sftune_uni)

    print("Naive Weighted Sampling...")
    set_seed(seed)
    ltrain_wght, lftune_wght, dev_wght, test_wght, strain_wght, sftune_wght = naive_sample(triples_to_freqs, indices_to_triples, testsize, dsize, ltsize, lftsize, stsize, sftsize, weight=True)
    overlaps = compute_overlaps(ltrain_wght, lftune_wght, strain_wght, sftune_wght, dev_wght, test_wght)
    illicitoverlaps = validate(ltrain_wght, lftune_wght, dev_wght, test_wght, strain_wght, sftune_wght)
    log(logger, language, "naive_wght", seed, ltrain_wght, lftune_wght, dev_wght, test_wght, strain_wght, sftune_wght, overlaps, illicitoverlaps)
    writesamples(os.path.join(outdir,"naive_weighted"), language, seed, ltrain_wght, lftune_wght, dev_wght, test_wght, strain_wght, sftune_wght)

    set_seed(seed)
    print("Train Feature in Test Overlap Aware Sampling...")
    print("Requested train-test feature overlap", round(foverlap*100,2))

    ltrain_fo, lftune_fo, dev_fo, test_fo, strain_fo, sftune_fo = nofreq_feataware_sample(feats_to_triples, triples, testsize, dsize, ltsize, lftsize, stsize, sftsize, foverlap)
    overlaps = compute_overlaps(ltrain_fo, lftune_fo, strain_fo, sftune_fo, dev_fo, test_fo)
    illicitoverlaps = validate(ltrain_fo, lftune_fo, dev_fo, test_fo, strain_fo, sftune_fo)
    log(logger, language, "featoverlap", seed, ltrain_fo, lftune_fo, dev_fo, test_fo, strain_fo, sftune_fo, overlaps, illicitoverlaps)
    writesamples(os.path.join(outdir,"overlap_aware"), language, seed, ltrain_fo, lftune_fo, dev_fo, test_fo, strain_fo, sftune_fo)

    close_log(logger)
    

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="make largetrain(smalltrain)/dev/test split with controlled smalltrain-test feature set overlap")
    parser.add_argument("freqfname", help="input file in UniMorph 3-column format, with 4th frequency column")
    parser.add_argument("rawfname", help="input file in UniMorph 3-column format")
    parser.add_argument("outdir", help="directory to write split files to")
    parser.add_argument("--small", type=int, help = "size of small training set")
    parser.add_argument("--large", type=int, help = "size of large training set")
    parser.add_argument("--smallfinetune", type=int, help = "size of finetuning subset of large training")
    parser.add_argument("--largefinetune", type=int, help = "size of finetuning subset of large training")
    parser.add_argument("--dev", type=int, help = "size of dev set")
    parser.add_argument("--test", type=int, help = "size of test set")
    parser.add_argument("--foverlap", type=float, help = "requested max train-test feature set overlap. Float in range [0,1]")
    parser.add_argument("--lang", help = "language to be written in output filenames")
    parser.add_argument("--seed", help = "random seed")
    args = parser.parse_args()
    

    main(args.freqfname, args.rawfname, args.outdir, args.small, args.large, args.smallfinetune, args.largefinetune, args.dev, args.test, args.foverlap, args.lang, args.seed)

