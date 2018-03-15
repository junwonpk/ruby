import datetime
import json
import math
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

def examinePredictions(filename, skip):
    print "Processing {}".format(filename)
    sid = SentimentIntensityAnalyzer()

    with open(filename, "r") as inFile:
        for i, line in enumerate(inFile, 1):
            if i <= skip:
                continue
            comment = json.loads(line)
            if (comment["num_child_comments"] == 0 and comment["prediction"] == 0) or \
               (comment["num_child_comments"] >= 1 and comment["prediction"] == 1):
                continue

            pss = sid.polarity_scores(comment['parent_comment']) 
            ss = sid.polarity_scores(comment['body']) 
            print ""
            print "-------------------------------------"
            print comment['parent_comment']
            print "-------------------------------------"
            print comment['body']
            print "-------------------------------------"
            print "Link: {} ID: {} Child Comments: {} Prediction: {} RT: {}".format(comment["link_id"], comment["id"], comment["num_child_comments"], comment["prediction"], comment["response_time_hours"])
            print "NeuNeg: {} NegPos: {}".format(
                    0.5 * (1 - abs(pss['compound'])) + ss['neg'],
                    pss['neg'] + ss['pos'])
            print pss
            print ss
            print "-------------------------------------"
            raw_input("")


if __name__ == "__main__":
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

            if i % 10000 == 0:
                print "Processed {} lines".format(i)
    stats.printAll()

    examinePredictions(sys.argv[1], 0)
