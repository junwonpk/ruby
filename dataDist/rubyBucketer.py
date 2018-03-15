# Ruby Dist.
# Given a discrete valued column
# Will split each value according to 0 and 1

import datetime
import json
import sys

def countColumn(filename, cName):
    print "Processing {}".format(filename)
    counts = {}
    with open(filename, "r") as inFile:
        for i, line in enumerate(inFile, 1):
            comment = json.loads(line)
            value = comment[cName]
            # created = datetime.datetime.fromtimestamp(
            #     comment["created_utc"]
            # )
            # value = created.time().hour

            if value not in counts:
                counts[value] = [0, 0]
            if comment["num_child_comments"] == 0:
                counts[value][0] += 1
            else:
                counts[value][1] += 1

            if i % 100000 == 0:
                print "Processed {} lines".format(i)
    return counts

def outputCounts(counts, filename):
    print "Outputting to {}".format(filename)
    with open(filename, "w") as outFile:
        for value, pair in counts.iteritems():
            outFile.write(str(value) + "," + str(pair[0]) + "," + str(pair[1]) + "\n")

if __name__ == "__main__":
    inFile = sys.argv[1]
    outFile = sys.argv[2]

    counts = countColumn(sys.argv[1], "depth")
    outputCounts(counts, sys.argv[2])
