import matplotlib.pyplot as plt
import json
import numpy as np
import nltk
import sys

# def plotHist(outFile, data):
#     plt.figure()
#     bins = 20
#     n, bins, patches = plt.hist(data, bins, facecolor='blue', alpha=0.5)
#     plt.savefig(outFile)

def plotCum(outFile, data):
    counts = []
    for datum in data:
        while datum >= len(counts):
            counts.append(0)
        counts[datum] += 1
    counts = np.array(counts)
    counts = counts / float(np.sum(counts))
    cdf = np.cumsum(counts)
    cdfToPlot = cdf[:501]

    plt.figure()
    plt.plot(range(0, len(cdfToPlot)), cdfToPlot, "b-", )
    plt.xlabel("Length")
    plt.ylabel("Fraction")
    plt.savefig(outFile)

    print "200:{} 250:{} 300:{} 350:{} 400:{} 500:{} 600:{} 750:{} 1000:{}".format(
        cdf[200],
        cdf[250],
        cdf[300],
        cdf[350],
        cdf[400],
        cdf[500],
        cdf[600],
        cdf[750],
        cdf[1000]
    )

def getLengthStats(filename, numComments):
    stats = {"lengths": []}
    with open(filename, "r") as inFile:
        for i, line in enumerate(inFile, 1):
            comment = json.loads(line)
            stats["lengths"].append(len(nltk.word_tokenize(comment["body"])))

            if i % 10000 == 0:
                print "Processed {} lines".format(i)

            if i == numComments:
                break

    return stats


if __name__ == "__main__":
    print "Loading Training Data"
    stats = getLengthStats(sys.argv[1] + "/Reddit2ndTrainTime", 100000)

    data = np.array(stats["lengths"])
    avg = sum(data) / float(len(data))
    std = np.sqrt(sum(data ** 2) / float(len(data)) - avg ** 2)
    print "Avg: {}, std: {}".format(avg, std)
    plotCum(sys.argv[2], stats["lengths"])
