import argparse
import collections
import json
import nltk
import numpy as np
import sys

# TODO: Uses too much RAM, can't go past 300,000 comments likely, will fix later
# TODO: Currently only supports a value of 1 for MIN_FREQ.
MIN_FREQ = 1
TOKEN_PAD = "TOKEN_PAD"
TOKEN_UNK = "TOKEN_UNK"

def process_comment(comment, vocab, frequencies):
    processedComment = []
    for word in nltk.word_tokenize(comment):
        word = process_word(word)
        if word not in vocab:
            vocab[word] = np.random.randn(len(vocab[TOKEN_PAD]))
            frequencies[word] = 0
        frequencies[word] += 1
        processedComment.append(word)
    return processedComment

def process_word(word):
    if 'http' in word:
        return 'TOKEN_HTTP_URL'
    if 'ftp' in word:
        return 'TOKEN_FTP_URL'
    if '@' in word:
        return 'TOKEN_AT_REFERENCE'
    word = word.lower()
    return word

def processComments(filename, numLines, vocab, frequencies):
    comments = []
    with open(filename, "r") as inFile:
        for i, line in enumerate(inFile, 1):
            if len(comments) >= numLines:
                break
            comment = json.loads(line)
            comment["body_t"] = process_comment(comment["body"], vocab, frequencies)
            comment["parent_comment_t"] = process_comment(comment["parent_comment"], vocab, frequencies)
            comments.append(comment)

            if i % 10000 == 0:
                print "Processed {} lines".format(i)

    return comments

def cleanFrequencies(vocab, frequencies):
    assert len(vocab) - 2 == len(frequencies)

    # Take care of special padding token.
    embed = [vocab[TOKEN_PAD], vocab[TOKEN_UNK]]
    vocab[TOKEN_PAD] = 0
    vocab[TOKEN_UNK] = 1

    # Loop through all words
    for word, count in frequencies.iteritems():
        if count < MIN_FREQ:
            del vocab[word]
            continue
        embed.append(vocab[word])
        vocab[word] = len(embed) - 1

    return vocab, np.asarray(embed)

def wordToIndex(word, vocab):
    if word in vocab:
        return vocab[word]
    return vocab[TOKEN_UNK]

def outputComments(comments, filename, vocab):
    with open(filename, "w") as outFile:
        for i, comment in enumerate(comments, 1):
            comment["body_t"] = [wordToIndex(word, vocab) for word in comment["body_t"]]
            comment["parent_comment_t"] = [wordToIndex(word, vocab) for word in comment["parent_comment_t"]]
            outFile.write(json.dumps(comment) + "\n")

            if i % 10000 == 0:
                print "Outputted {} lines".format(i)

def outputVocab(vocab, filename):
    vocabList = [TOKEN_UNK] * len(embed)
    for word, index in vocab.iteritems():
        vocabList[index] = word
    with open(filename, "w") as outFile:
        for word in vocabList:
            outFile.write(word.encode('utf-8') + "\n")

# Builds a vocab.
def loadWordVectors(inFilename):
    print "Loading word vectors"
    embedSize = 0
    vocab = {}
    frequencies = {}
    with open(inFilename, 'r') as inFile:
        for i, line in enumerate(inFile, 1):
            row = line.strip().split(' ')
            vocab[row[0]] = np.array([float(num) for num in row[1:]])
            frequencies[row[0]] = 0
            embedSize = len(row) - 1

            if i % 100000 == 0:
                print "Processed {} lines".format(i)
    vocab[TOKEN_PAD] = np.zeros(embedSize)
    vocab[TOKEN_UNK] = np.random.randn(embedSize)
    print "Loaded {} words".format(len(vocab))

    return vocab, frequencies

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert train/dev/test into word embeddings and vocab")
    parser.add_argument("-w", "--wordVectors", help="word vector file name", required=True)
    parser.add_argument("-i", "--inDir", help="Input directory of train/dev/test", required=True)
    parser.add_argument("-o", "--outDir", help="Output directory", required=True)
    parser.add_argument("-t", "--test", help="Whether or not to include test", type=bool, default=False)
    parser.add_argument("-n", "--numLines", help="Number of lines of train/dev/test to process", type=int, default=-1)
    args = parser.parse_args()

    if args.test:
        print "Test set not supported yet"
        sys.exit(1)
    numLines = args.numLines if args.numLines >= 0 else float('inf')

    vocab, frequencies = loadWordVectors(args.wordVectors)

    print "Processing Training Data"
    trainComments = processComments(
        args.inDir + "/Reddit2ndTrainTime",
        numLines,
        vocab,
        frequencies)

    print "Processing Dev Data"
    devComments = processComments(
        args.inDir + "/Reddit2ndDevTime",
        numLines,
        vocab,
        frequencies)

    print "Cleaning frequencies"
    vocab, embed = cleanFrequencies(vocab, frequencies)
    assert len(vocab) == len(embed)
    print "Vocab size: {}".format(len(vocab))

    print "Outputting train comments"
    outputComments(trainComments, args.outDir + "/Reddit2ndTrainT", vocab)

    print "Outputting dev comments"
    outputComments(devComments, args.outDir + "/Reddit2ndDevT", vocab)

    print "Outputting embeddings"
    np.savetxt(args.outDir + "/embed.txt", embed)

    print "Outputting vocab"
    outputVocab(vocab, args.outDir + "/vocab.txt")
