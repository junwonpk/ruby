# Ruby Prep 10.
# Sentiment Analysis using Vader

import json
import sys
from nltk.sentiment.vader import SentimentIntensityAnalyzer

def annotate(inFilename, outFilename):
    print "Input: {}".format(inFilename)
    print "Output: {}".format(outFilename)

    sid = SentimentIntensityAnalyzer()
    with open(inFilename, "r") as inFile, \
         open(outFilename, "w") as outFile:
        for i, line in enumerate(inFile, 1):
            comment = json.loads(line)
            ss = sid.polarity_scores(comment["body"])
            comment["positive"] = ss["pos"]
            comment["neutral"] = ss["neu"]
            comment["negative"] = ss["neg"]
            comment["compound"] = ss["compound"]

            pss = sid.polarity_scores(comment["parent_comment"])
            comment["ppositive"] = pss["pos"]
            comment["pneutral"] = pss["neu"]
            comment["pnegative"] = pss["neg"]
            comment["pcompound"] = pss["compound"]

            outFile.write(json.dumps(comment) + "\n")

            if i % 10000 == 0:
                print "Processed {} lines".format(i)

if __name__ == "__main__":
    annotate(sys.argv[1], sys.argv[2])
