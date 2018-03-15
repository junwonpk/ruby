# Ruby Final Baseline.
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
            solver="sag",
            max_iter=200,
            verbose=True)
    lr.fit(trainX, trainLabels)
    trainPreds = lr.predict(trainX)
    print "Train Accuracy: {}".format(accuracy_score(trainLabels, trainPreds))
    devPreds = lr.predict(devX)
    print "Dev Accuracy: {}".format(accuracy_score(devLabels, devPreds))
    print "Matrix: \n{}".format(confusion_matrix(devLabels, devPreds))


def loadComments(filename, numComments):
    # Load comments.
    comments = []
    labels = []
    with open(filename, "r") as inFile:
        for line in inFile:
            if len(comments) >= numComments:
                break
            comment = json.loads(line)
            if comment["num_child_comments"] == 0:
                bucket = 0
            else:
                bucket = 1
            comments.append({
                "i": comment["id"],
                "b": comment["body"]
            })
            labels.append(bucket)
    return comments, labels

if __name__ == "__main__":
    print "Loading Training Data"
    trainComments, trainLabels = loadComments(
        sys.argv[1] + "/ProcessedTrain",
        200000)

    print "Loading Dev Data"
    devComments, devLabels = loadComments(
        sys.argv[1] + "/ProcessedDev",
        40000)

    softmaxRegression(trainComments, trainLabels, devComments, devLabels)
