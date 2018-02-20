# Ruby Prep 8.
# Moves comments to correct file

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

def writeComments(inFilename, links, outFilename):
    print "Writing from {} to {} with {} links".format(inFilename, outFilename, len(links))

    with open(inFilename, "r") as inFile, \
         open(outFilename, "w") as outFile:
        for i, line in enumerate(inFile, 1):
            comment = json.loads(line)
            commentId = comment["id"]
            if commentId in links:
                outFile.write(json.dumps(comment) + "\n")

            if i % 100000 == 0:
                print "Processed {} lines".format(i)

if __name__ == "__main__":
    # links = loadLinks(sys.argv[1] + "/Reddit2ndTrain")
    # writeComments(sys.argv[2] + "/RedditAll", links, sys.argv[3] + "/Reddit2ndTrain")

    # links = loadLinks(sys.argv[1] + "/Reddit2ndDev")
    # writeComments(sys.argv[2] + "/RedditAll", links, sys.argv[3] + "/Reddit2ndDev")

    links = loadLinks(sys.argv[1] + "/Reddit2ndTest")
    writeComments(sys.argv[2] + "/RedditAll", links, sys.argv[3] + "/Reddit2ndTest")
