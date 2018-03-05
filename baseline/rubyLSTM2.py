import matplotlib.pyplot as plt
import json
import numpy as np
import nltk
import sys
import tensorflow as tf

# TODO: Make this more accurate, current is 2191.
MAXDOCLENGTH = 250

# TODO: Turning on and off features
ADDRT = True
ADDTIME = False
ADDLEN = True
ADDF = ADDRT or ADDTIME or ADDLEN

def doWork(embed, trainComments, trainCommentfs, trainLabels, devComments, devCommentfs, devLabels, saveOut, saveIn):
    # Constants.
    vocabSize = len(embed)
    embedDim = len(embed[0])
    batchSize = len(trainComments[0])

    # model parameters
    # learningRates = [0.01] * 10 + [0.005] * 10 + [0.003] * 10 + [0.002] * 10
    learningRates = [0.01] * 3 + [0.005] * 2 + [0.003] * 5 + [0.002] * 10 + [0.001] * 5 + [0.0005] * 5 + [0.0004] * 5 + [0.0003] * 5 + [0.0002] * 5 + [0.0001] * 5
    lstmUnits = 64
    dropoutKeepProb = 0.9
    numClasses = 2
    epochs = len(learningRates)
    layer2Units = 32

    # Additional comment features.
    numCommentfs = len(trainCommentfs[0][0])

    # Print info:
    print (
        "vocabSize: {}, embedDim: {}, batchSize: {}, numTrain: {}, numDev: {}, " +
        "maxDocLength: {}, \nlearningRates: {}, \nlstmUnits: {}, dropoutKeepProb: {}, epochs: {}, " +
        "ADDRT: {}, ADDTIME: {}, ADDLEN: {}, additionalFeatures: {}, layer2Units: {}"
    ).format(
        vocabSize,
        embedDim,
        batchSize,
        batchSize * len(trainComments),
        batchSize * len(devComments),
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
    x = tf.placeholder(tf.int32, [batchSize, 2 * MAXDOCLENGTH])
    y = tf.placeholder(tf.float32, [batchSize, numClasses])
    commentfs = tf.placeholder(tf.float32, [batchSize, numCommentfs])
    keepProbVar = tf.placeholder(tf.float32)
    learningRate = tf.placeholder(tf.float32)
    
    # Create embedding tranform.
    with tf.name_scope("embedding"):
        W = tf.Variable(tf.constant(0.0, shape=[vocabSize, embedDim]), trainable=False, name="W")
        embedding_placeholder = tf.placeholder(tf.float32, [vocabSize, embedDim])
        embedding_init = W.assign(embedding_placeholder)
        data = tf.nn.embedding_lookup(W, x)

    # LSTM
    lstmCell = tf.contrib.rnn.BasicLSTMCell(lstmUnits)
    lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=keepProbVar)
    value, _ = tf.nn.dynamic_rnn(lstmCell, data, dtype=tf.float32)
     
    # Output to pred
    # weight = tf.get_variable("weight", shape=[lstmUnits, numClasses], initializer=tf.contrib.layers.xavier_initializer())
    weight = tf.Variable(tf.truncated_normal([lstmUnits + numCommentfs, layer2Units]))
    bias = tf.Variable(tf.constant(0.1, shape=[layer2Units]))
    value = tf.transpose(value, [1, 0, 2])
    last = tf.gather(value, int(value.get_shape()[0]) - 1)

    # Edit to account for additional features.
    layer1Input = last
    if ADDF:
        layer1Input = tf.concat([last, commentfs], axis=1)

    # relu on layer 1, add additional layer
    layer2Input = tf.nn.relu(tf.matmul(layer1Input, weight) + bias)
    with tf.name_scope("layer2"):
        layer2W = tf.Variable(tf.truncated_normal([layer2Units, numClasses]))
        layer2b = tf.Variable(tf.constant(0.1, shape=[numClasses]))
    prediction = tf.matmul(layer2Input, layer2W) + layer2b

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
        sess.run(embedding_init, feed_dict={embedding_placeholder: embed})
    
        for i in range(epochs):
            epochLoss = 0
            epochAccuracy = 0
            for batchNum in range(len(trainComments)):
                commentsBatch = trainComments[batchNum]
                labelsBatch = trainLabels[batchNum]
                feedDict = {x: commentsBatch, y: labelsBatch, keepProbVar: dropoutKeepProb}
                feedDict[learningRate] = learningRates[i]
                if ADDF:
                    feedDict[commentfs] = trainCommentfs[batchNum]
                batchAccuracy, batchLoss, _ = sess.run(
                    [accuracy, loss, optimizer],
                    feedDict)
                epochLoss += batchLoss
                epochAccuracy += batchAccuracy
                if (batchNum + 1) % 100 == 0:
                    print "Epoch: {}, Batch: {}".format(i+1, batchNum+1)
            indLoss = epochLoss / float(len(trainComments))
            indAccuracy = epochAccuracy / float(len(trainComments))
            losses.append(indLoss)
            trainAccuracies.append(indAccuracy)
            print "Epoch: {}, Loss: {}, Accuracy: {}".format(i+1, indLoss, indAccuracy)

            devAccuracy = 0
            for batchNum in range(len(devComments)):
                commentsBatch = devComments[batchNum]
                labelsBatch = devLabels[batchNum]
                feedDict = {x: commentsBatch, y: labelsBatch, keepProbVar: 1.0}
                if ADDF:
                    feedDict[commentfs] = devCommentfs[batchNum]
                devAccuracy += sess.run(accuracy, feedDict)
            indDevAccuracy = devAccuracy / float(len(devComments))
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

# Words from 1 to vocab.
# All words not in vocab should be mapped to index 0.
def loadWordVectors(inFilename):
    print "Loading word vectors"
    vocab = {}
    embed = [[]]
    with open(inFilename, 'r') as inFile:
        for i, line in enumerate(inFile, 1):
            row = line.strip().split(' ')
            vocab[row[0]] = i
            embed.append([float(num) for num in row[1:]])

            if i % 100000 == 0:
                print "Processed {} lines".format(i)
    embed[0] = np.zeros(len(embed[1]))
    print "Loaded {} words".format(len(vocab))

    return vocab, np.asarray(embed)

def wordToIndex(word, vocab):
    word = word.lower()
    if word in vocab:
        return vocab[word]
    return 0

def pad(a, i):
    if len(a) > i:
        return a[0:i]
    return a + [0] * (i-len(a))

def loadComments(filename, vocab, numBatches, batchSize):
    comments = []
    commentfs = []
    labels = []
    with open(filename, "r") as inFile:
        commentsBatch = []
        commentfsBatch = []
        labelsBatch = []
        for i, line in enumerate(inFile, 1):
            if len(comments) >= numBatches:
                break
            comment = json.loads(line)
            parentTokenized = [wordToIndex(word, vocab) for word in nltk.word_tokenize(comment["parent_comment"])]
            commentTokenized = [wordToIndex(word, vocab) for word in nltk.word_tokenize(comment["body"])]
            finalCommentInput = pad(parentTokenized, MAXDOCLENGTH)
            finalCommentInput.extend(pad(commentTokenized, MAXDOCLENGTH))
            commentsBatch.append(finalCommentInput)

            additionalFeatures = []
            if ADDRT:
                additionalFeatures.append(comment["response_time_hours"])
            if ADDTIME:
                additionalFeatures.append(comment["time_of_day"])
                additionalFeatures.append(comment["weekday"])
            if ADDLEN:
                additionalFeatures.append(len(commentTokenized))
            commentfsBatch.append(additionalFeatures)

            if comment["num_child_comments"] == 0:
                labelsBatch.append([1, 0])
            else:
                labelsBatch.append([0, 1])

            if len(commentsBatch) == batchSize:
                comments.append(np.asarray(commentsBatch))
                commentfs.append(np.asarray(commentfsBatch))
                labels.append(np.asarray(labelsBatch))
                commentsBatch = []
                commentfsBatch = []
                labelsBatch = []

            if i % 10000 == 0:
                print "Processed {} lines".format(i)

    return comments, commentfs, labels

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
    # About 6.6 GB RAM for 100d.
    wordVectorFilename = "fileName"
    vocab, embed = loadWordVectors(wordVectorFilename)
    batchSize = 200

    print "Loading Training Data"
    trainComments, trainCommentfs, trainLabels = loadComments(
        sys.argv[1] + "/Reddit2ndTrainTime",
        vocab,
        500,
        batchSize)

    print "Loading Dev Data"
    devComments, devCommentfs, devLabels = loadComments(
        sys.argv[1] + "/Reddit2ndDevTime",
        vocab,
        100,
        batchSize)

    losses, trainAccuracies, devAccuracies =  doWork(
        embed,
        trainComments,
        trainCommentfs,
        trainLabels,
        devComments,
        devCommentfs,
        devLabels,
        None,
        None
    )
    plot(losses, trainAccuracies, devAccuracies, sys.argv[2])
