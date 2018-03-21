# Ruby Prep 11.
# Depth of comments

import json
import sys

def loadLinks(filename):
    print "Loading links from {}".format(filename)
    links = {}
    with open(filename, "r") as inFile:
        for line in inFile:
            link = json.loads(line)
            links[link["i"]] = link

    return links

def annotateDepth(links):
    print "Annotating Depth"
    for i, linkId in enumerate(links, 1):
        link = links[linkId]
        parentId = link["p"]
        if "d" in link:
            continue

        depth = 0
        stack = []
        while not parentId.startswith("t3"):
            if "d" in link:
                depth = link["d"]
                break
            if parentId[3:] not in links:
                depth = -1000000
                break

            stack.append(link)
            linkId = parentId[3:]
            link = links[linkId]
            parentId = link["p"]
        stack.append(link)

        while len(stack) > 0:
            link = stack.pop()
            link["d"] = depth
            depth += 1

        if i % 100000 == 0:
            print "{} of {} links".format(i, len(links))

    print "Detecting problems"
    counter = 0
    for linkId in links:
        link = links[linkId]
        if link["d"] < 0:
            counter += 1
    print "{} problems".format(counter)

def outputLinks(links, outFilename):
    print "Outputting links to {}".format(outFilename)
    with open(outFilename, 'w') as outFile:
        for _, link in links.iteritems():
            outFile.write(json.dumps(link) + "\n")


def annotateComments(inFilename, outFilename, links):
    print "Input: {}".format(inFilename)
    print "Output: {}".format(outFilename)

    problems = 0
    with open(inFilename, "r") as inFile, \
         open(outFilename, "w") as outFile:
        for i, line in enumerate(inFile, 1):
            comment = json.loads(line)
            comment["depth"] = links[comment["id"]]["d"]
            if comment["depth"] < 0:
                problems += 1

            outFile.write(json.dumps(comment) + "\n")

            if i % 10000 == 0:
                print "Processed {} lines".format(i)
    print "{} problems".format(problems)

if __name__ == "__main__":
    phase1 = False
    phase2 = True

    if phase1:
        links = loadLinks(sys.argv[1] + "/RedditAll")
        annotateDepth(links)
        outputLinks(links, sys.argv[2] + "/RedditAll")

    if phase2:
        links = loadLinks(sys.argv[1] + "/RedditAll")
        annotateComments(sys.argv[2], sys.argv[3], links)
