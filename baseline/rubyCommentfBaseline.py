# Ruby RT Baseline
# How well does response time predict further responses?
import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import sys

def softmaxRegression(trainComments, trainLabels, devComments, devLabels):
    trainX = np.array(trainComments)
    devX = np.array(devComments)

    print "Starting Regression"
    lr = LogisticRegression(
            C=1.0,
            # class_weight="balanced",
            solver="sag",
            max_iter=1000,
            # multi_class="multinomial",
            verbose=True)
    lr.fit(trainX, trainLabels)
    trainPreds = lr.predict(trainX)
    print "Train Accuracy: {}".format(accuracy_score(trainLabels, trainPreds))
    devPreds = lr.predict(devX)
    print "Dev Accuracy: {}".format(accuracy_score(devLabels, devPreds))
    print "Matrix: \n{}".format(confusion_matrix(devLabels, devPreds))


def loadComments(filename, numComments, capBuckets=False):
    # Create bucket lookup.
    # bucketLookup = [0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3]
    bucketLookup = [0]
    while len(bucketLookup) < 1780:
        bucketLookup.append(1)

    # Bucket cap.
    bucketCounts = [0 for i in range(5)]
    if capBuckets:
        bucketCap = numComments / 5
    else:
        bucketCap = float('inf')

    # Load comments.
    comments = []
    labels = []
    with open(filename, "r") as inFile:
        for line in inFile:
            if len(comments) >= numComments:
                break
            comment = json.loads(line)
            bucket = bucketLookup[comment["num_child_comments"]]
            if bucketCounts[bucket] >= bucketCap:
                continue
            bucketCounts[bucket] += 1

            commentfs = []
            # TODO: Do stuff here to test different features
            commentfs.append(comment["response_time_hours"])
            commentfs.append(abs(comment["score"]))
            # commentfs.append(comment["time_of_day"])
            # commentfs.append(comment["weekday"])

            # END
            comments.append(commentfs)

            labels.append(bucket)

    return comments, labels

if __name__ == "__main__":
    print "Loading Training Data"
    trainComments, trainLabels = loadComments(
        sys.argv[1] + "/Reddit2ndTrainTime",
        10000)

    print "Loading Dev Data"
    devComments, devLabels = loadComments(
        sys.argv[1] + "/Reddit2ndDevTime",
        2000)

    softmaxRegression(trainComments, trainLabels, devComments, devLabels)
