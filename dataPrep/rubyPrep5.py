# Ruby Prep 5.
# Splits into train-dev-test.

import json
import random

def loadLinks(filename):
    print "Loading links from {}".format(filename)
    links = {}
    with open(filename, "r") as inFile:
        for line in inFile:
            link = json.loads(line)
            links[link["i"]] = link

    return links

def splitData(inFilename, outTrainName, outDevName, outTestName):
    print "Input: {}".format(inFilename)
    print "Train: {}".format(outTrainName)
    print "Dev: {}".format(outDevName)
    print "Test: {}".format(outTestName)

    # 80-10-10
    probTrain = 0.8
    probDev = 0.9

    with open(inFilename, "r") as inFile, \
         open(outTrainName, "w") as outTrain, \
         open(outDevName, "w") as outDev, \
         open(outTestName, "w") as outTest:
        for i, line in enumerate(inFile, 1):
            randNum = random.random()
            if randNum < probTrain:
                outTrain.write(line)
            elif randNum < probDev:
                outDev.write(line)
            else:
                outTest.write(line)

            if i % 100000 == 0:
                print "Processed {} lines".format(i)

if __name__ == "__main__":
    splitData(
        "/media/vvayfarer/ExtraDrive1/bigData/news2/RedditAll",
        "/media/vvayfarer/ExtraDrive1/bigData/redditComments/RedditTrain",
        "/media/vvayfarer/ExtraDrive1/bigData/redditComments/RedditDev",
        "/media/vvayfarer/ExtraDrive1/bigData/redditComments/RedditTest"
    )
