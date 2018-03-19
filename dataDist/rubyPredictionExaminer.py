import collections
import datetime
import json
import math
import numpy as np
import sys
from nltk.sentiment.vader import SentimentIntensityAnalyzer

class Stats():
    def __init__(self):
        self.numT = 0
        self.numC = 0
        self.numI = 0
        self.lengthC = 0
        self.lengthI = 0
        self.lengthBucketsC = [0] * 5
        self.lengthBucketsI = [0] * 5
        self.timeBucketsC = [0] * 24
        self.timeBucketsI = [0] * 24
        self.rtBucketsC = [0] * 13
        self.rtBucketsI = [0] * 13
        self.childBucketsC = [0] * 6
        self.childBucketsI = [0] * 6
        self.sitBuckets0 = [0] * 10
        self.sitBuckets1 = [0] * 10
        self.pnBuckets0 = [0] * 10
        self.pnBuckets1 = [0] * 10

        self.sid = SentimentIntensityAnalyzer()

    def printAll(self):
        print "Correct %: {}".format(self.numC / float(self.numT))
        print ""
        print "Avg length of correct: {}".format(self.lengthC / float(self.numC))
        print "Avg length of incorrect: {}".format(self.lengthI / float(self.numI))
        print "Length correct % [<50, 100, 150, 200, 200+]:\n {}".format(self.getPrintingBuckets(self.lengthBucketsC, self.lengthBucketsI))
        print ""
        print "Time Hour correct % :\n {}".format(self.getPrintingBuckets(self.timeBucketsC, self.timeBucketsI))
        print ""
        print "RT correct % [0.25, 0.5...9.5, 24]:\n {}".format(self.getPrintingBuckets(self.rtBucketsC, self.rtBucketsI))
        print "RT totals :\n {}".format(self.getTotalBuckets(self.rtBucketsC, self.rtBucketsI))
        print ""
        print "Num Children correct [0, 1, 2, 6, 14]:\n {}".format(self.getPrintingBuckets(self.childBucketsC, self.childBucketsI))
        print "Num Children totals:\n {}".format(self.getTotalBuckets(self.childBucketsC, self.childBucketsI))
        print ""
        print "SIT 0: \n {}".format(self.sitBuckets0)
        print "SIT 1: \n {}".format(self.sitBuckets1)
        print "PN 0: \n {}".format(self.pnBuckets0)
        print "PN 1: \n {}".format(self.pnBuckets1)

    def addLength(self, length, correct):
        buckets = self.lengthBucketsC
        if correct:
            self.lengthC += length
        if not correct:
            self.lengthI += length
            buckets = self.lengthBucketsI
        if length < 200:
            buckets[length / 50] += 1
        else:
            buckets[-1] += 1

    def addTime(self, time, correct):
        buckets = self.timeBucketsC
        if not correct:
            buckets = self.timeBucketsI
        created = datetime.datetime.fromtimestamp(time)
        buckets[created.time().hour] += 1

    def addRT(self, rt, correct):
        buckets = self.rtBucketsC
        if not correct:
            buckets = self.rtBucketsI
        if rt < 0.5:
            buckets[int(rt / 0.25)] += 1
        elif rt >= 0.5 and rt < 9.5:
            buckets[int((rt + 0.5) / 1) + 1] += 1
        elif rt < 24:
            buckets[-2] += 1
        else:
            buckets[-1] += 1

    def addChild(self, numChildComments, correct):
        buckets = self.childBucketsC
        if not correct:
            buckets = self.childBucketsI
        cutoffs = [0, 1, 2, 6, 14, float('inf')]
        for i in range(len(cutoffs)):
            if numChildComments <= cutoffs[i]:
                buckets[i] += 1
                break

    def addSIT(self, parent, body, numChildComments):
        ss = self.sid.polarity_scores(body) 
        pss = self.sid.polarity_scores(parent) 
        buckets = self.sitBuckets0
        pnBuckets = self.pnBuckets0
        if numChildComments > 0:
            buckets = self.sitBuckets1
            pnBuckets = self.pnBuckets1
        cutoffs = [-0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1.1]
        for i in range(len(cutoffs)):
            if ss["compound"] <= cutoffs[i]:
                buckets[i] += 1
                break
        cutoffs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        for i in range(len(cutoffs)):
            # score = max(0.5 * (1 - abs(pss['compound'])) + ss['neg'], 0.5 * pss['neg'] + ss['pos'])
            # score = 0.5 * pss['neg'] + ss['pos']
            # score = 0.5 * pss['pos'] + ss['neg']
            score= abs(pss['pos'] - ss['pos'])
            if score <= cutoffs[i]:
                pnBuckets[i] += 1
                break


    def getPrintingBuckets(self, bucketsC, bucketsI):
        buckets = []
        for i in range(len(bucketsC)):
            buckets.append(bucketsC[i] / float(bucketsC[i] + bucketsI[i]))
        return buckets
    
    def getTotalBuckets(self, bucketsC, bucketsI):
        buckets = []
        for i in range(len(bucketsC)):
            buckets.append(bucketsC[i] + bucketsI[i])
        return buckets

def examinePredictions(filename, skip, vocab=None, allAlphas=None):
    print "Processing {}".format(filename)

    with open(filename, "r") as inFile:
        for i, line in enumerate(inFile, 1):
            if i <= skip:
                continue
            comment = json.loads(line)
            # if (comment["num_child_comments"] == 0 and comment["prediction"] == 0) or \
            #    (comment["num_child_comments"] >= 1 and comment["prediction"] == 1):
            #     continue
            alphas = allAlphas[i -1]
            top5 = reversed(sorted(range(len(alphas)), key=lambda j: alphas[j])[-5:])
            top5words = [vocab[comment['body_t'][j]] for j in top5 if j < len(comment['body_t'])]

            print ""
            print "-------------------------------------"
            print comment['parent_comment']
            print "-------------------------------------"
            print comment['body']
            print "-------------------------------------"
            print "Link: {} ID: {} Child Comments: {} Prediction: {} RT: {}".format(comment["link_id"], comment["id"], comment["num_child_comments"], comment["prediction"], comment["response_time_hours"])
            print top5words
            print "-------------------------------------"
            raw_input("")

def sortAndPrintTop5(words):
    wordList = []
    for word in words:
        wordList.append((word, words[word]))
    wordList.sort(key=lambda pair: pair[1])
    print list(reversed(wordList[-30:]))
    print sum([pair[1] for pair in wordList])

def calcTop5(filename, vocab, allAlphas):
    print "Processing {}".format(filename)

    blacklist = ["?", ";", "s", "you", "m", "that", "if", "is", "re", "``", "this", "they", "TOKEN_UNK", "TOKEN_PAD", ""]

    wordsa0p0 = collections.defaultdict(int)
    wordsa0p1 = collections.defaultdict(int)
    wordsa1p0 = collections.defaultdict(int)
    wordsa1p1 = collections.defaultdict(int)

    with open(filename, "r") as inFile:
        for i, line in enumerate(inFile):
            comment = json.loads(line)

            words = None
            if comment["num_child_comments"] == 0:
                if comment["prediction"] == 0:
                    words = wordsa0p0
                else:
                    words = wordsa0p1
            else:
                if comment["prediction"] == 0:
                    words = wordsa1p0
                else:
                    words = wordsa1p1

            alphas = allAlphas[i]
            sortedA = reversed(sorted(range(len(alphas)), key=lambda j: alphas[j]))
            word = None
            for index in sortedA:
                if index >= len(comment["body_t"]):
                    continue
                vocabIndex = comment["body_t"][index]
                if vocab[vocabIndex] in blacklist:
                    continue
                word = vocab[index]
                break
            if word is not None:
                words[word] += 1


            # top5words = [vocab[comment['body_t'][j]] for j in top5 if j < len(comment['body_t'])]
            # for word in top5words:
            #     words[word] += 1

            if (i + 1) % 10000 == 0:
                print "Processed {} lines".format(i + 1)

    print "Actual 0, predicted 0"
    sortAndPrintTop5(wordsa0p0)
    print "Actual 0, predicted 1"
    sortAndPrintTop5(wordsa0p1)
    print "Actual 1, predicted 0"
    sortAndPrintTop5(wordsa1p0)
    print "Actual 1, predicted 1"
    sortAndPrintTop5(wordsa1p1)

if __name__ == "__main__":
    ca0p0 = []
    ca0p1 = []
    ca1p0 = []
    ca1p1 = []

    stats = Stats()
    with open(sys.argv[1], "r") as inFile:
        for i, line in enumerate(inFile, 1):
            comment = json.loads(line)
            if (comment["num_child_comments"] == 0 and comment["prediction"] == 0) or \
               (comment["num_child_comments"] >= 1 and comment["prediction"] == 1):
                stats.numC += 1
                stats.addLength(len(comment["body_t"]), True)
                stats.addTime(comment["created_utc"], True)
                stats.addRT(comment["response_time_hours"], True)
                stats.addChild(comment["num_child_comments"], True)
            else:
                stats.numI += 1
                stats.addLength(len(comment["body_t"]), False)
                stats.addTime(comment["created_utc"], False)
                stats.addRT(comment["response_time_hours"], False)
                stats.addChild(comment["num_child_comments"], False)
                stats.addSIT(comment["parent_comment"], comment["body"], comment["num_child_comments"])
            stats.numT += 1

            cList = None
            if comment["num_child_comments"] == 0:
                if comment["prediction"] == 0:
                    cList = ca0p0
                else:
                    cList = ca0p1
            else:
                if comment["prediction"] == 0:
                    cList = ca1p0
                else:
                    cList = ca1p1

            weight = max(comment["soft"])
            newpair = (weight, comment["body"])
            if len(cList) < 5:
                cList.append(newpair)
            elif weight > cList[-1][0]:
                cList.append(newpair)
                cList.sort(key=lambda p: p[0], reverse=True)
                cList.pop()

            if i % 10000 == 0:
                print "Processed {} lines".format(i)
    stats.printAll()

    print "Actual 0, predicted 0"
    for thing in ca0p0:
        print thing
    print "Actual 0, predicted 1"
    for thing in ca0p1:
        print thing
    print "Actual 1, predicted 0"
    for thing in ca1p0:
        print thing
    print "Actual 1, predicted 1"
    for thing in ca1p1:
        print thing


    # print "Loading Vocab"
    # vocab = []
    # with open(sys.argv[1]) as inFile:
    #     for line in inFile:
    #         vocab.append(line.strip())

    # print "Loading alphas"
    # allAlphas = np.loadtxt(sys.argv[2])
    # 
    # calcTop5(sys.argv[3], vocab, allAlphas)
    # examinePredictions(sys.argv[3], 0, vocab, allAlphas)
