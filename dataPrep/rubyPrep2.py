# Ruby Prep 2.
# Extracts link information.

import json
import sys

def toLinks(inFilename, outFilename):
    print "Processing {} to {}".format(inFilename, outFilename)
    with open(inFilename, "r") as inFile, open(outFilename, "w") as outFile:
        for i, line in enumerate(inFile, 1):
            comment = json.loads(line)
            link = {"i": comment["id"], "p": comment["parent_id"]}
            outFile.write(json.dumps(link) + "\n")
            if i % 100000 == 0:
                print "Processed {} lines".format(i)

if __name__ == "__main__":
    toLinks(
        "/media/vvayfarer/ExtraDrive1/bigData/news1/" + sys.argv[1],
        "/media/vvayfarer/ExtraDrive1/bigData/links1/" + sys.argv[1],
    )
