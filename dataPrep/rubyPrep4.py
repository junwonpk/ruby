# Ruby Prep 4.
# Annotates comments with num child comments, removes deleted comments.

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

def annotateLines(links, inFilename, outFilename):
    print "Processing {} to {}".format(inFilename, outFilename)
    with open(inFilename, "r") as inFile, open(outFilename, "w") as outFile:
        for i, line in enumerate(inFile, 1):
            comment = json.loads(line)
            if comment["body"].startswith("[removed]") or comment["body"].startswith("[deleted]"):
                continue
            comment["num_child_comments"] = links[comment["id"]]["c"]
            outFile.write(json.dumps(comment) + "\n")
            if i % 100000 == 0:
                print "Processed {} lines".format(i)


if __name__ == "__main__":
    links = loadLinks("/media/vvayfarer/ExtraDrive1/bigData/links2/" + sys.argv[1])
    print "Loaded {} links".format(len(links))
    annotateLines(
        links,
        "/media/vvayfarer/ExtraDrive1/bigData/news1/" + sys.argv[2],
        "/media/vvayfarer/ExtraDrive1/bigData/news2/" + sys.argv[2],
    )
