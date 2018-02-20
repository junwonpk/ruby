# Ruby Dist.
# Counts integer columns.

import json
import sys


def countColumn(filename, cName):
    print "Processing {}".format(filename)
    counts = []
    with open(filename, "r") as inFile:
        for i, line in enumerate(inFile, 1):
            comment = json.loads(line)
            count = comment[cName]
            while count >= len(counts):
                counts.append(0)
            counts[count] += 1
            if i % 100000 == 0:
                print "Processed {} lines".format(i)
    return counts

def outputCounts(counts, filename):
    print "Outputting to {}".format(filename)
    with open(filename, "w") as outFile:
        for i, count in enumerate(counts):
            outFile.write(str(i) + "," + str(count) + "\n")

if __name__ == "__main__":
    inFile = sys.argv[1]
    outFile = sys.argv[2]

    counts = countColumn(sys.argv[1], "num_child_comments")
    outputCounts(counts, sys.argv[2])
