import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import datetime

def get_minibatches(data, minibatch_size, shuffle=True):
    """
    Iterates through the provided data one minibatch at at time. You can use this function to
    iterate through data in minibatches as follows:

        for inputs_minibatch in get_minibatches(inputs, minibatch_size):
            ...

    Or with multiple data sources:

        for inputs_minibatch, labels_minibatch in get_minibatches([inputs, labels], minibatch_size):
            ...

    Args:
        data: there are two possible values:
            - a list or numpy array
            - a list where each element is either a list or numpy array
        minibatch_size: the maximum number of items in a minibatch
        shuffle: whether to randomize the order of returned data
    Returns:
        minibatches: the return value depends on data:
            - If data is a list/array it yields the next minibatch of data.
            - If data a list of lists/arrays it returns the next minibatch of each element in the
              list. This can be used to iterate through multiple data sources
              (e.g., features and labels) at the same time.

    """
    list_data = type(data) is list and (type(data[0]) is list or type(data[0]) is np.ndarray)
    data_size = len(data[0]) if list_data else len(data)
    indices = np.arange(data_size)
    if shuffle:
        np.random.shuffle(indices)
    for minibatch_start in np.arange(0, data_size, minibatch_size):
        minibatch_indices = indices[minibatch_start:minibatch_start + minibatch_size]
        yield [minibatch(d, minibatch_indices) for d in data] if list_data \
            else minibatch(data, minibatch_indices)

def minibatch(data, minibatch_idx):
    return data[minibatch_idx] if type(data) is np.ndarray else [data[i] for i in minibatch_idx]

def pad(a, i):
    mask = [1] * len(a)
    if len(a) > i:
        return a[:i], mask[:i]
    padding = i - len(a)
    return a + [0] * padding, mask + [0] * padding

def loadComments(filename, maxComments, config):
    comments = []
    masks = []
    commentps = []
    maskps = []
    commentfs = []
    labels = []
    with open(filename, "r") as inFile:
        for i, line in enumerate(inFile, 1):
            if len(comments) >= maxComments:
                break
            comment = json.loads(line)

            commentInput, maskInput = pad(comment["body_t"], config["maxDocLength"])
            comments.append(commentInput)
            masks.append(maskInput)

            commentpInput, maskpInput= pad(comment["parent_comment_t"], config["maxDocLength"])
            commentps.append(commentpInput)
            maskps.append(maskpInput)

            commentf = []
            if config["addRT"]:
                commentf.append(comment["response_time_hours"])
            if config["addTime"]:
                commentf.append(comment["time_of_day"])
                commentf.append(comment["weekday"])
            if config["addTime2"]:
                created = datetime.datetime.fromtimestamp(
                    comment["created_utc"]
                )
                value = created.time().hour
                timeVec = [0] * 24
                timeVec[value] = 1
                commentf.extend(timeVec)
            if config["addLength"]:
                commentf.append(len(comment["body_t"]))
            commentfs.append(commentf)

            if comment["num_child_comments"] == 0:
                labels.append([1, 0])
            else:
                labels.append([0, 1])

            if i % 10000 == 0:
                print "Processed {} lines".format(i)

    return [comments, masks, commentps, maskps, commentfs, labels]


def labelCommentsWithPredictions(inFilename, outFilename, predictions):
    with open(inFilename, "r") as inFile, open(outFilename, "w") as outFile:
        for i, line in enumerate(inFile):
            if i >= len(predictions):
                break
            comment = json.loads(line)
            comment["prediction"] = predictions[i]
            outFile.write(json.dumps(comment) + "\n")

            if (i + 1) % 10000 == 0:
                print "Processed {} lines".format(i)

def printConfig(config):
    print "-----------------------------------------"
    print ["{}: {}".format(k, v) for k, v in sorted(config.iteritems())]
    print "-----------------------------------------"

def plot(losses, trainAccuracies, devAccuracies, outputFile):
    xs = range(1, len(losses) + 1)
    plt.figure()
    plt.plot(xs, losses, "r-", label="loss")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.savefig(outputFile + "1.png")

    plt.figure()
    trainAcc, = plt.plot(xs, trainAccuracies, "b-", label="trainAcc")
    devAcc, = plt.plot(xs, devAccuracies, "g-", label="devAcc")
    plt.xlabel("epochs")
    plt.legend(handles=[trainAcc, devAcc])
    plt.savefig(outputFile + "2.png")


