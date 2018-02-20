# Ruby Plot.
# Plots y v x of two columns

import matplotlib.pyplot as plt
import numpy as np
import json
import sys

def getXY(filename, xName, yName):
    print "Processing {}".format(filename)
    xs = []
    ys = []
    with open(filename, "r") as inFile:
        for i, line in enumerate(inFile, 1):
            comment = json.loads(line)
            if comment[xName] > 48:
                continue
            if comment[yName] > 400:
                continue
            xs.append(comment[xName])
            ys.append(comment[yName])
            if i % 100000 == 0:
                print "Processed {} lines".format(i)
    return xs, ys

def plotXY(xs, ys, outputTitle, outputFile):
    plt.figure()
    plt.plot(xs, ys, "b+", label="xy1")
    plt.xlabel("response time hours")
    plt.ylabel("child comments")
    plt.title(outputTitle)
    plt.savefig(outputFile)

    # c1, = plt.plot(range(1, 21), c1Errors, "b+", label="c1")
    # c2, = plt.plot(range(1, 21), c2Errors, "r+", label="c2")
    # plt.legend(handles=[c1, c2])


def outputCounts(counts, filename):
    print "Outputting to {}".format(filename)
    with open(filename, "w") as outFile:
        for i, count in enumerate(counts):
            outFile.write(str(i) + "," + str(count) + "\n")

if __name__ == "__main__":
    inFile = sys.argv[1]
    outFile = sys.argv[2]

    xs, ys = getXY(inFile, "response_time_hours", "num_child_comments")
    points = plotXY(xs, ys, "child comments vs. response time", outFile)
