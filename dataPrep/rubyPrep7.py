# Ruby Prep 7
# Given links splits into 2nd and top comments
# Splits snd comments into train, dev, test
# Removes parents of snd comments in train from dev and test

import json
import random
import sys

def loadLinks(filename):
    print "Loading links from {}".format(filename)
    links = {}
    with open(filename, "r") as inFile:
        for line in inFile:
            link = json.loads(line)
            links[link["i"]] = link

    return links

def split2ndTop(links, topOutFilename):
    print "Spltting into 2nd and top links"
    sndLinks = []
    with open(topOutFilename, "w") as topOutFile:
        for i, (linkId, link) in enumerate(links.iteritems(), 1):
            parentId = link["p"]
            if parentId.startswith("t3"):
                topOutFile.write(json.dumps(link) + "\n")
                continue
            else:
                parentId = parentId[3:]
                if parentId in links:
                    sndLinks.append(link)

            if i % 100000 == 0:
                print "Processed {} links".format(i)
    return sndLinks

def splitData(inLinks):
    print "Splitting in Train, Dev, Test"

    # Output
    outTrain = []
    outDev = []
    outTest = []

    # 80-10-10
    probTrain = 0.8
    probDev = 0.9

    for i, link in enumerate(inLinks, 1):
        randNum = random.random()
        if randNum < probTrain:
            outTrain.append(link)
        elif randNum < probDev:
            outDev.append(link)
        else:
            outTest.append(link)

        if i % 100000 == 0:
            print "Processed {} lines".format(i)
    return outTrain, outDev, outTest

def removeParents(links):
    print "Shuffling links"
    random.shuffle(links)

    print "Making sure no parents of comments tested are not tested"
    parentCache = {}
    linkCache = {}
    newLinks = []
    for i, link in enumerate(links):
        if link["c"] == 0 and random.randint(1, 100) > 52:
            continue
        if link["c"] == 1 and random.randint(1, 100) > 75:
            continue
        if link["c"] == 2 and random.randint(1, 100) > 90:
            continue
        linkId = link["i"]
        parentId = link["p"][3:]
        if linkId in parentCache or parentId in linkCache:
            continue
        linkCache[linkId] = 1
        parentCache[parentId] = 1
        newLinks.append(link)

        if i % 100000 == 0:
            print "Processed {} links".format(i)

    return newLinks

# def removeParents(train, dev, test):
#     print "Removing parents of train from dev and test"
# 
#     for i, (_, link) in enumerate(train.iteritems(), 1):
#         parentId = link["p"]
#         parentId = parentId[3:]
#         if parentId in dev:
#             del dev[parentId]
#         if parentId in test:
#             del test[parentId]
# 
#         if i % 100000 == 0:
#             print "Processed {} lines".format(i)

def writeToFile(links, outFilename):
    print "Writing {} links to {}".format(len(links), outFilename)
    with open(outFilename, "w") as outFile:
        for i, link in enumerate(links, 1):
            outFile.write(json.dumps(link) + "\n")

            if i % 100000 == 0:
                print "Processed {} links".format(i)

if __name__ == "__main__":
    links = loadLinks(sys.argv[1] + "/RedditAll")
    sndLinks = split2ndTop(links, sys.argv[1] + "/RedditTop")
    sndLinks = removeParents(sndLinks)
    train, dev, test = splitData(sndLinks)
    writeToFile(train, sys.argv[1] + "/Reddit2ndTrain")
    writeToFile(dev, sys.argv[1] + "/Reddit2ndDev")
    writeToFile(test, sys.argv[1] + "/Reddit2ndTest")
