import argparse
import json
import nltk
import numpy as np
import re

# TODO: Uses too much RAM, can't go past 300,000 comments likely, will fix later
MIN_FREQ = 2
TOKEN_PAD = "TOKEN_PAD"
TOKEN_UNK = "TOKEN_UNK"

def process_comment(comment, prevocab, newvocab, frequencies):
    processedComment = []
    for word in nltk.word_tokenize(comment):
        word = process_word(word)
        processedComment.append(word)

        if word in prevocab:
            frequencies[word] += MIN_FREQ
        elif newvocab is None:
            continue
        elif word in newvocab:
            frequencies[word] += 1
        else:
            newvocab[word] = np.random.randn(len(prevocab[TOKEN_PAD]))
            frequencies[word] = 1
    return processedComment

def process_word(word):
    # Take care of special symbols by themselves.
    if word == "*" or word == "=" or word == ".":
        return word
    if word == ".." or word == "...":
        return "..."
    if word == "'":
        return word

    # Special link words.
    http = re.compile('/*/')
    if 'ftp' in word:
        return 'TOKEN_FTP_URL'
    if '@' in word:
        return 'TOKEN_AT_REFERENCE'
    if 'http' in word or http.search(word) or '=' in word:
        return 'TOKEN_HTTP_URL'

    # Numbers.
    digits = re.compile('\d')
    if digits.search(word):
        return 'TOKEN_NUMBER'

    # Other things.
    if word.startswith("'"):
        word = word[1:]
    if word.endswith("'"):
        word = word[:-1]

    word = word.lower().replace("*", "").replace(".", "")
    return word

def processComments(filename, numLines, prevocab, newvocab, frequencies):
    comments = []
    with open(filename, "r") as inFile:
        for i, line in enumerate(inFile, 1):
            if len(comments) >= numLines:
                break
            comment = json.loads(line)
            comment["body_t"] = process_comment(comment["body"], prevocab, newvocab, frequencies)
            comment["parent_comment_t"] = process_comment(comment["parent_comment"], prevocab, newvocab, frequencies)
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

    return np.asarray(embed)

def wordToIndex(word, vocab):
    if word in vocab:
        return vocab[word]
    return vocab[TOKEN_UNK]

def outputComments(comments, filename, vocab):
    unknownCounter = 0
    with open(filename, "w") as outFile:
        for i, comment in enumerate(comments, 1):
            comment["body_t"] = [wordToIndex(word, vocab) for word in comment["body_t"]]
            comment["parent_comment_t"] = [wordToIndex(word, vocab) for word in comment["parent_comment_t"]]
            outFile.write(json.dumps(comment) + "\n")

            for wordNum in comment["body_t"]:
                if wordNum == vocab[TOKEN_UNK]:
                    unknownCounter += 1
            for wordNum in comment["parent_comment_t"]:
                if wordNum == vocab[TOKEN_UNK]:
                    unknownCounter += 1

            if i % 10000 == 0:
                print "Outputted {} lines".format(i)
    print "Encountered {} unknown words".format(unknownCounter)

def outputVocab(vocab, filename):
    vocabList = [TOKEN_UNK] * len(embed)
    for word, index in vocab.iteritems():
        vocabList[index] = word
    with open(filename, "w") as outFile:
        for word in vocabList:
            outFile.write(word.encode('utf-8') + "\n")

def outputNewWords(newWords, filename):
    with open(filename, "w") as outFile:
        for word in newWords:
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

    numLines = args.numLines if args.numLines >= 0 else float('inf')
    prevocab, frequencies = loadWordVectors(args.wordVectors)
    newvocab = {}

    print "Processing Training Data"
    trainComments = processComments(
        args.inDir + "/Reddit2ndTrainTime",
        numLines,
        prevocab,
        newvocab,
        frequencies)
    print "Potential new vocab: {}".format(len(newvocab))

    print "Processing Dev Data"
    devComments = processComments(
        args.inDir + "/Reddit2ndDevTime",
        numLines,
        prevocab,
        None,
        frequencies)

    if args.test:
        print "Processing Test Data"
        testComments = processComments(
            args.inDir + "/Reddit2ndTestTime",
            numLines,
            prevocab,
            None,
            frequencies)

    print "Cleaning frequencies"
    for word in newvocab:
        assert word not in prevocab
        prevocab[word] = newvocab[word]
    newvocab = None
    embed = cleanFrequencies(prevocab, frequencies)
    vocab = prevocab
    assert len(vocab) == len(embed)
    print "Vocab size: {}".format(len(vocab))

    print "Outputting train comments"
    outputComments(trainComments, args.outDir + "/Reddit2ndTrainT", vocab)

    print "Outputting dev comments"
    outputComments(devComments, args.outDir + "/Reddit2ndDevT", vocab)

    if args.test:
        print "Outputting test comments"
        outputComments(testComments, args.outDir + "/Reddit2ndTestT", vocab)

    print "Outputting embeddings"
    np.savetxt(args.outDir + "/embed.txt", embed)

    print "Outputting vocab"
    outputVocab(vocab, args.outDir + "/vocab.txt")
