# Ruby Baseline 3.
# Ruby Softmax on n-gram baseline for annotated data set.
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import sys

def softmaxRegression(trainComments, trainLabels, devComments, devLabels):
    corpus = []
    for comment in trainComments:
        corpus.append(comment['b'])
    for comment in devComments:
        corpus.append(comment['b'])

    print "Extracting ngrams"
    cv = CountVectorizer(ngram_range=(1, 2))
    X = cv.fit_transform(corpus)
    trainX = X[:len(trainComments)]
    devX = X[len(trainComments):]

    print "Starting Regression"
    lr = LogisticRegression(
            C=1.0,
            # class_weight="balanced",
            solver="sag",
            max_iter=200,
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
            comments.append({
                "i": comment["id"],
                "b": comment["body"]
            })
            labels.append(bucket)

    return comments, labels

if __name__ == "__main__":
    print "Loading Training Data"
    trainComments, trainLabels = loadComments(
        sys.argv[1] + "/Reddit2ndTrain",
        1000000)

    print "Loading Dev Data"
    devComments, devLabels = loadComments(
        sys.argv[1] + "/Reddit2ndDev",
        100000)

    softmaxRegression(trainComments, trainLabels, devComments, devLabels)

