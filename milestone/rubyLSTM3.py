import argparse
import matplotlib.pyplot as plt
import json
import numpy as np
import tensorflow as tf

# TODO: Make this more accurate, current is 2191.
MAXDOCLENGTH = 250
BATCHSIZE = 256

# TODO: Turning on and off features
ADDRT = True
ADDTIME = False
ADDLEN = True
ADDF = ADDRT or ADDTIME or ADDLEN

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

def doWork(embed, trainData, devData, saveOut, saveIn):
    # Constants.
    vocabSize = len(embed)
    embedDim = len(embed[0])
    numTrain = len(trainData[0])
    numDev = len(devData[0])

    # model parameters
    # learningRates = [0.01] * 10 + [0.005] * 10 + [0.003] * 10 + [0.002] * 10
    learningRates = [0.01] * 3 + [0.005] * 2 + [0.003] * 5 + [0.002] * 10 + [0.001] * 5 + [0.0005] * 5
    # learningRates = [0.01] * 3 + [0.005] * 2 + [0.003] * 5 + [0.002] * 10 + [0.001] * 5 + [0.0005] * 5 + [0.0004] * 5 + [0.0003] * 5 + [0.0002] * 5 + [0.0001] * 5
    # learningRates = [0.01] * 10 + [0.005] * 10
    lstmUnits = 64
    dropoutKeepProb = 0.9
    numClasses = 2
    epochs = len(learningRates)
    layer2Units = 32

    # Additional comment features.
    numCommentfs = len(trainData[2][0])

    # Print info:
    print (
        "vocabSize: {}, embedDim: {}, BATCHSIZE: {}, numTrain: {}, numDev: {}, " +
        "maxDocLength: {}, \nlearningRates: {}, \nlstmUnits: {}, dropoutKeepProb: {}, epochs: {}, " +
        "ADDRT: {}, ADDTIME: {}, ADDLEN: {}, additionalFeatures: {}, layer2Units: {}"
    ).format(
        vocabSize,
        embedDim,
        BATCHSIZE,
        numTrain,
        numDev,
        MAXDOCLENGTH,
        learningRates,
        lstmUnits,
        dropoutKeepProb,
        epochs,
        ADDRT,
        ADDTIME,
        ADDLEN,
        numCommentfs,
        layer2Units
    )

    # create input placeholders
    x = tf.placeholder(tf.int32, [None, MAXDOCLENGTH])
    y = tf.placeholder(tf.float32, [None, numClasses])
    commentfs = tf.placeholder(tf.float32, [None, numCommentfs])
    masks = tf.placeholder(tf.float32, [None, MAXDOCLENGTH])
    keepProbVar = tf.placeholder(tf.float32)
    learningRate = tf.placeholder(tf.float32)
    
    # Create embedding tranform.
    with tf.name_scope("embedding"):
        E = tf.get_variable("E", initializer=embed, trainable=False)
        embeddings = tf.nn.embedding_lookup(E, x)

    # LSTM
    lstmCell = tf.contrib.rnn.BasicLSTMCell(lstmUnits)
    lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=keepProbVar)
    cellOutputs, _ = tf.nn.dynamic_rnn(lstmCell, embeddings, dtype=tf.float32)
     
    # Output to pred
    cellOutputs = tf.transpose(cellOutputs, [2, 0, 1]) # cells, batches, len
    maskedOutputs = tf.reduce_sum(cellOutputs * masks, axis=2) / tf.reduce_sum(masks, axis=1)
    layer1Input = tf.transpose(maskedOutputs, [1, 0]) # batches, cells
    if ADDF:
        layer1Input = tf.concat([layer1Input, commentfs], axis=1)

    # relu on layer 1
    W1 = tf.get_variable("W1", shape=[lstmUnits + numCommentfs, layer2Units], initializer=tf.initializers.truncated_normal())
    b1 = tf.get_variable("b1", shape=[layer2Units], initializer=tf.constant_initializer(0.1))
    layer2Input = tf.nn.relu(tf.matmul(layer1Input, W1) + b1)

    # softmax on layer 2
    with tf.name_scope("layer2"):
        W2 = tf.get_variable("W2", shape=[layer2Units, numClasses], initializer=tf.initializers.truncated_normal())
        b2 = tf.get_variable("b2", shape=[numClasses], initializer=tf.constant_initializer(0.1))
    prediction = tf.matmul(layer2Input, W2) + b2

    # Pred transform
    correctPred = tf.equal(tf.argmax(prediction,1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))

    # Loss and optimizer
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learningRate).minimize(loss)

    # Saver
    saver = tf.train.Saver()

    # Collect Info.
    losses = []
    trainAccuracies = []
    devAccuracies = []

    with tf.Session() as sess:
        # Variable Initialization.
        if saveIn:
            saver.restore(sess, saveIn)
        else:
            sess.run(tf.global_variables_initializer())
    
        for epoch in range(epochs):
            epochLoss = 0
            epochAccuracy = 0
            for batchNum, trainBatches in enumerate(get_minibatches(trainData, BATCHSIZE)):
                commentsBatch = trainBatches[0]
                commentfsBatch = trainBatches[1]
                labelsBatch = trainBatches[2]
                maskBatch = trainBatches[3]
                batchSize = len(commentsBatch)

                feedDict = {x: commentsBatch, y: labelsBatch, keepProbVar: dropoutKeepProb}
                feedDict[masks] = maskBatch
                feedDict[learningRate] = learningRates[epoch]
                if ADDF:
                    feedDict[commentfs] = commentfsBatch

                batchAccuracy, batchLoss, _ = sess.run([accuracy, loss, optimizer], feedDict)
                epochLoss += batchLoss * batchSize
                epochAccuracy += batchAccuracy * batchSize
                if (batchNum + 1) % 100 == 0:
                    print "Epoch: {}, Batch: {}".format(epoch + 1, batchNum + 1)
            indLoss = epochLoss / float(numTrain)
            indAccuracy = epochAccuracy / float(numTrain)
            losses.append(indLoss)
            trainAccuracies.append(indAccuracy)
            print "Epoch: {}, Loss: {}, Accuracy: {}".format(epoch + 1, indLoss, indAccuracy)

            devAccuracy = 0
            for batchNum, devBatches in enumerate(get_minibatches(devData, BATCHSIZE)):
                commentsBatch = devBatches[0]
                commentfsBatch = devBatches[1]
                labelsBatch = devBatches[2]
                maskBatch = devBatches[3]
                batchSize = len(commentsBatch)

                feedDict = {x: commentsBatch, y: labelsBatch, keepProbVar: 1.0}
                feedDict[masks] = maskBatch
                if ADDF:
                    feedDict[commentfs] = commentfsBatch
                devAccuracy += sess.run(accuracy, feedDict) * batchSize
            indDevAccuracy = devAccuracy / float(numDev)
            devAccuracies.append(indDevAccuracy)
            print "Dev Accuracy: {}".format(indDevAccuracy)

            # savePath = saver.save(sess, saveOut)
            # print "Model saved at {}".format(savePath)

    # Print out summary.
    bestDevAccuracy = 0
    bestIndex = 0
    for i, accuracy in enumerate(devAccuracies):
        if accuracy > bestDevAccuracy:
            bestDevAccuracy = accuracy
            bestIndex = i
    print "Best Dev of {} at epoch {}, train acc: {}, train loss: {}".format(
        bestDevAccuracy,
        bestIndex + 1, 
        trainAccuracies[bestIndex],
        losses[bestIndex])

    # Return series.
    return losses, trainAccuracies, devAccuracies

def pad(a, i):
    mask = [1] * len(a)
    if len(a) > i:
        return a[:i], mask[:i]
    padding = i - len(a)
    return a + [0] * padding, mask + [0] * padding

def loadComments(filename, maxComments):
    comments = []
    commentfs = []
    labels = []
    masks = []
    with open(filename, "r") as inFile:
        for i, line in enumerate(inFile, 1):
            if len(comments) >= maxComments:
                break
            comment = json.loads(line)
            # commentInput = pad(comment["parent_comment_t"], MAXDOCLENGTH)
            commentInput, maskInput = pad(comment["body_t"], MAXDOCLENGTH)
            comments.append(commentInput)
            masks.append(maskInput)

            additionalFeatures = []
            if ADDRT:
                additionalFeatures.append(comment["response_time_hours"])
            if ADDTIME:
                additionalFeatures.append(comment["time_of_day"])
                additionalFeatures.append(comment["weekday"])
            if ADDLEN:
                additionalFeatures.append(len(comment["body_t"]))
            commentfs.append(additionalFeatures)

            if comment["num_child_comments"] == 0:
                labels.append([1, 0])
            else:
                labels.append([0, 1])

            if i % 10000 == 0:
                print "Processed {} lines".format(i)

    # IMPORTANT
    return [comments, commentfs, labels, masks]

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LSTM model")
    parser.add_argument("-i", "--inDir", help="Input directory of train/dev/test", required=True)
    parser.add_argument("-o", "--outPrefix", help="Output prefix for plot", required=True)
    args = parser.parse_args()

    embed = np.loadtxt(args.inDir + "/embed.txt", dtype=np.float32)

    print "Loading Training Data"
    trainData = loadComments(
        args.inDir + "/Reddit2ndTrainT",
        100000)

    print "Loading Dev Data"
    devData = loadComments(
        args.inDir + "/Reddit2ndDevT",
        20000)

    losses, trainAccuracies, devAccuracies =  doWork(
        embed,
        trainData,
        devData,
        None,
        None
    )
    plot(losses, trainAccuracies, devAccuracies, args.outPrefix)
