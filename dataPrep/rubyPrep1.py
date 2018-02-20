# Ruby Prep 1.
# Outputs only news.

import json
import sys

def filterLines(inFilename, outFilename, key, value):
    print "Processing {} to {}".format(inFilename, outFilename)
    with open(inFilename, "r") as inFile, open(outFilename, "w") as outFile:
        for i, line in enumerate(inFile, 1):
            comment = json.loads(line)
            if comment[key] == value:
                outFile.write(json.dumps(comment) + "\n")
            if i % 100000 == 0:
                print "Processed {} lines".format(i)

if __name__ == "__main__":
    filterLines(
        "/media/vvayfarer/ExtraDrive1/bigData/jpIn/" + sys.argv[1],
        "/media/vvayfarer/ExtraDrive1/bigData/news1/" + sys.argv[1],
        "subreddit",
        "news"
    )
