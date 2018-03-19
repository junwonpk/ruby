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
            # if comment[xName] > 48:
            #     continue
            # if comment[yName] > 400:
            #     continue
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

def bucketizePlot(xs, ys, outputFile):
    zeros = []
    ones = []
    for x, y in zip(xs, ys):
        if y == 0:
            zeros.append[x]
        else:
            ones.append[x]
    zeros = np.array(zeros)
    n, bins, patches = plt.hist(zeros, bins, facecolor='blue', alpha=0.5)

    plt.xlabel("response time hours")
    plt.ylabel("percentage")
    plt.savefig(outFile)


def plotBar1():
    plt.figure(figsize=(4, 4))

    x = [10, 20, 50, 100]
    y = [0.650025, 0.651225, 0.64825, 0.64895]
    index = np.arange(len(x)) / float(10)
    barWidth = 0.06


    rects1 = plt.bar(
        index,
        y,
        barWidth,
        alpha=0.8,
        color='b',
        label='a')
    plt.xlabel("Number of Filters")
    plt.ylabel("Dev Accuracy")

    axes = plt.gca()
    axes.set_ylim([0.63, 0.66])
    axes.grid(True)
    axes.set_axisbelow(True)

    plt.xticks(index, x)
    plt.tight_layout()
    plt.savefig("../statsout/finalReport/hp1.png")

def plotBar2():
    plt.figure(figsize=(9, 4))

    x = ["[1, 2]",
         "[2, 3]",
         "[3, 4]",
         "[4, 5]",
         "[1, 2, 3]",
         "[2, 3, 4]",
         "[3, 4, 5]",
         "[1, 2, 3, 4]",
         "[2, 3, 4, 5]",
         "[1, 2, 3, 4, 5]"]
    y = [0.650725, 0.650275, 0.650275, 0.648925, 0.65005, 0.64915, 0.648625, 0.651425, 0.649625, 0.6485]


    index = np.arange(len(x)) / float(10)
    barWidth = 0.08


    rects1 = plt.bar(
        index[:4],
        y[:4],
        barWidth,
        alpha=0.8,
        color='b',
        label='a')
    rects1 = plt.bar(
        index[4:7],
        y[4:7],
        barWidth,
        alpha=0.8,
        color='r',
        label='a')
    rects1 = plt.bar(
        index[7:9],
        y[7:9],
        barWidth,
        alpha=0.8,
        color='b',
        label='a')
    rects1 = plt.bar(
        index[9:10],
        y[9:10],
        barWidth,
        alpha=0.8,
        color='r',
        label='a')

    plt.xlabel("Filter Sizes")
    plt.ylabel("Dev Accuracy")

    axes = plt.gca()
    axes.set_ylim([0.63, 0.66])
    axes.grid(True)
    axes.set_axisbelow(True)

    plt.xticks(index, x)
    plt.tight_layout()
    plt.savefig("../statsout/finalReport/hp2.png")

def plotBar3():
    plt.figure(figsize=(4, 4))

    x = [16, 32, 64, 128]
    y = [0.646575, 0.641125, 0.651125, 0.648825]
    index = np.arange(len(x)) / float(10)
    barWidth = 0.06


    rects1 = plt.bar(
        index,
        y,
        barWidth,
        alpha=0.8,
        color='b',
        label='a')
    plt.xlabel("Batch Sizes")
    plt.ylabel("Dev Accuracy")

    axes = plt.gca()
    axes.set_ylim([0.63, 0.66])
    axes.grid(True)
    axes.set_axisbelow(True)

    plt.xticks(index, x)
    plt.tight_layout()
    plt.savefig("../statsout/finalReport/hp3.png")

def plotBar4():
    plt.figure(figsize=(4, 4))

    x = ["[0,0]", "[1,1]", "[2,2]", "[3,6]", "[7,14]", "[15, inf]"]
    y = [0.5771677960261858, 0.6208689428703743, 0.6970168795781875, 0.7544401880550236, 0.8120015561174869, 0.8364108194430582]
    index = np.arange(len(x)) / float(10)
    barWidth = 0.06

    rects1 = plt.bar(
        index,
        y,
        barWidth,
        alpha=0.8,
        color='b',
        label='a')
    plt.xlabel("Number of Descendant Comments")
    plt.ylabel("Fraction of Correct Predictions")

    axes = plt.gca()
    # axes.set_ylim([0.63, 0.66])
    axes.grid(True)
    axes.set_axisbelow(True)

    plt.xticks(index, x)
    plt.tight_layout()
    plt.savefig("../statsout/finalReport/bar4.png")



def outputCounts(counts, filename):
    print "Outputting to {}".format(filename)
    with open(filename, "w") as outFile:
        for i, count in enumerate(counts):
            outFile.write(str(i) + "," + str(count) + "\n")

if __name__ == "__main__":
    # inFile = sys.argv[1]
    # outFile = sys.argv[2]

    # xs, ys = getXY(inFile, "response_time_hours", "num_child_comments")
    # points = plotXY(xs, ys, "child comments vs. response time", outFile)

    # plotBar1()
    # plotBar2()
    # plotBar3()
    plotBar4()
