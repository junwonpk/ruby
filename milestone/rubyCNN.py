import argparse
import numpy as np
import tensorflow as tf
import time

import utils

# IMPORTANT: Contains the configurations for the LSTM
def getConfig():
    config = {
        "maxDocLength": 250,  # Max is 2191
        "batchSize": 64,
        "addRT": True,
        "addTime": False,
        "addTime2": False,
        "addLength": True,
        "addSIT": True,
        "addCommentp": False,
        "addDepth": False
    }
    config["addCommentf"] = config["addRT"] or config["addTime"] or config["addLength"] or config["addTime2"]
    config["learningRates"] = [0.005] * 2 + [0.003] * 2 + [0.002] * 2 + [0.001] * 2 + [0.0005] * 2
    config["dropoutKeepProb"] = 0.9
    config["numTrain"] = 200000
    config["numDev"] = 40000
    config["numEpochs"] = len(config["learningRates"])
    config["predictScore"] = False
    config["filterSizes"] = [1, 2, 3]
    config["numFilters"] = 100
    config["layer2Units"] = 64
    config["lambda"] = None

    # Junk.
    # config["learningRates"] = [0.01] * 5 + [0.005] * 5 + [0.003] * 5 + [0.002] * 5 + [0.001] * 5 + [0.0005] * 5 + [0.0003] * 5 + [0.0001] * 5
    # config["learningRates"] = [0.01] * 5 + [0.005] * 5 + [0.003] * 5 + [0.002] * 5 + [0.001] * 5 + [0.0005] * 5 + [0.0003] * 5 + [0.0001] * 5
    # config["learningRates"] = [0.01] * 3 + [0.005] * 2 + [0.003] * 5 + [0.002] * 10 + [0.001] * 5 + [0.0005] * 5 + [0.0003] * 5 + [0.0001] * 5
    # config["learningRates"] = [0.01] * 3 + [0.005] * 2 + [0.003] * 5 + [0.002] * 10 + [0.001] * 5 + [0.0005] * 5
    # learningRates = [0.01] * 10 + [0.005] * 10 + [0.003] * 10 + [0.002] * 10
    # learningRates = [0.01] * 3 + [0.005] * 2 + [0.003] * 5 + [0.002] * 10 + [0.001] * 5 + [0.0005] * 5 + [0.0004] * 5 + [0.0003] * 5 + [0.0002] * 5 + [0.0001] * 5
    # learningRates = [0.01] * 10 + [0.005] * 10
    # config["layer3Units"] = 32

    return config

def getCNNOutputs(embeddings, dropoutKeepProb, scope, config):
    # CNN
    pooledOutputs = []
    for filterSize in config["filterSizes"]:
        with tf.variable_scope("conv-maxpool-{}-{}".format(scope, filterSize)):
            # Convolution Layer
            Wc = tf.get_variable(
                "Wc",
                shape=[filterSize, config["embedDim"], 1, config["numFilters"]],
                initializer=tf.initializers.truncated_normal(stddev=0.1))
            bc = tf.get_variable(
                "biasc",
                shape=[config["numFilters"]],
                initializer=tf.constant_initializer(0.1))
            conv = tf.nn.conv2d(
                embeddings,
                Wc,
                strides=[1, 1, 1, 1],
                padding="VALID",
                name="conv")
            # Apply nonlinearity
            h = tf.nn.relu(tf.nn.bias_add(conv, bc), name="relu")
            # Max-pooling over the outputs
            pooled = tf.nn.max_pool(
                h,
                ksize=[1, config["maxDocLength"] - filterSize + 1, 1, 1],
                strides=[1, 1, 1, 1],
                padding="VALID",
                name="pool")
            pooledOutputs.append(pooled)
 
    # Combine all the pooled features
    numFiltersTotal = config["numFilters"] * len(config["filterSizes"])
    h_pool = tf.concat(pooledOutputs, axis=3)
    h_pool_flat = tf.reshape(h_pool, [-1, numFiltersTotal])

    return h_pool_flat

def train(embed, trainData, devData, config):
    # Create input placeholders
    comments = tf.placeholder(tf.int32, [None, config["maxDocLength"]])
    masks = tf.placeholder(tf.float32, [None, config["maxDocLength"]])
    commentps = tf.placeholder(tf.int32, [None, config["maxDocLength"]])
    maskps = tf.placeholder(tf.float32, [None, config["maxDocLength"]])
    commentfs = tf.placeholder(tf.float32, [None, config["numCommentfs"]])
    labels = tf.placeholder(tf.float32, [None, config["numClasses"]])
    dropoutKeepProb = tf.placeholder(tf.float32)
    learningRate = tf.placeholder(tf.float32)

    # Create embedding tranform.
    with tf.variable_scope("embedding"):
        E = tf.get_variable("E", initializer=embed, trainable=False)
        embeddings = tf.expand_dims(tf.nn.embedding_lookup(E, comments), -1)
        embeddingps = tf.expand_dims(tf.nn.embedding_lookup(E, commentps), -1)

    # CNN
    cnnOutputs = getCNNOutputs(embeddings, dropoutKeepProb, "c", config)
    numFiltersTotal = config["numFilters"] * len(config["filterSizes"])
    if config["addCommentp"]:
        cnnOutputps = getCNNOutputs(embeddingps, dropoutKeepProb, "p", config)
        S = tf.get_variable(
            "S",
            shape=[numFiltersTotal, numFiltersTotal],
            initializer=tf.initializers.truncated_normal(stddev=0.1))
        cnnOutputps = tf.expand_dims(tf.reduce_sum(tf.matmul(cnnOutputs, S) * cnnOutputps, axis=1), -1)
        cnnOutputs = tf.concat([cnnOutputs, cnnOutputps], axis=1)
        numFiltersTotal += 1

    # Dropout
    hDroutput = tf.nn.dropout(cnnOutputs, dropoutKeepProb)

    # Add in extra features
    if config["addCommentf"]:
        hDroutput = tf.concat([hDroutput, commentfs], axis=1)

    # Layer 1 ReLu
    with tf.variable_scope("layer1"):
        W1 = tf.get_variable(
            "W1",
            shape=[numFiltersTotal + config["numCommentfs"], config["layer2Units"]],
            initializer=tf.initializers.truncated_normal(stddev=0.1))
        b1 = tf.get_variable(
            "bias1",
            shape=[config["layer2Units"]], 
            initializer=tf.constant_initializer(0.1))
        layer1Output = tf.nn.relu(tf.matmul(hDroutput, W1) + b1)

    # Dropout
    layer1Droutput = tf.nn.dropout(layer1Output, dropoutKeepProb)

    # Output
    with tf.variable_scope("layer2"):
        W2 = tf.get_variable(
            "W2",
            shape=[config["layer2Units"], config["numClasses"]],
            initializer=tf.initializers.truncated_normal(stddev=0.1))
        b2 = tf.get_variable(
            "bias2",
            shape=[config["numClasses"]],
            initializer=tf.constant_initializer(0.1))
    prediction = tf.matmul(layer1Droutput, W2) + b2

    # Accuracy
    prediction2 = tf.argmax(prediction, 1)
    labels2 = tf.argmax(labels, 1)
    confusionMatrix = tf.confusion_matrix(labels2, prediction2)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction2, labels2), tf.float32))

    # Loss and optimizer
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels))
    if config["lambda"]:
        l2 = config["lambda"] * sum(
            tf.nn.l2_loss(variable) for
            variable in
            tf.trainable_variables()
            if not ("bias" in variable.name))
        loss += l2
    optimizer = tf.train.AdamOptimizer(learning_rate=learningRate).minimize(loss)

    # Saver
    saver = tf.train.Saver()

    # Collect Info.
    losses = []
    trainAccuracies = []
    devAccuracies = []

    # Collect best info.
    bestDevIndex = None
    bestDevPredictions = []
    bestDevConfusionMatrix = []

    with tf.Session() as sess:
        # Variable Initialization.
        if config["restoreDir"]:
            print "Restoring from {}".format(config["restoreDir"])
            saver.restore(sess, config["restoreDir"])
        else:
            sess.run(tf.global_variables_initializer())

        for epoch in range(config["numEpochs"]):
            start = time.time()

            # Training.
            epochLoss = 0
            epochAccuracy = 0
            for batchNum, batches in enumerate(utils.get_minibatches(trainData, config["batchSize"])):
                feedDict = {
                    comments: batches[0],
                    masks: batches[1],
                    labels: batches[5],
                    learningRate: config["learningRates"][epoch],
                    dropoutKeepProb: config["dropoutKeepProb"]
                }
                if config["addCommentp"]:
                    feedDict[commentps] = batches[2]
                    feedDict[maskps] = batches[3]
                if config["addCommentf"]:
                    feedDict[commentfs] = batches[4]

                batchSize = len(batches[0])
                batchAccuracy, batchLoss, _ = sess.run([accuracy, loss, optimizer], feedDict)
                epochLoss += batchLoss * batchSize
                epochAccuracy += batchAccuracy * batchSize
                if (batchNum + 1) % 100 == 0:
                    print "Epoch: {}, Batch: {}".format(epoch + 1, batchNum + 1)
            losses.append(epochLoss / float(config["numTrain"]))
            trainAccuracies.append(epochAccuracy / float(config["numTrain"]))
            print "Epoch: {}, Loss: {}, Accuracy: {}".format(epoch + 1, losses[-1], trainAccuracies[-1])

            # Dev.
            epochPredictions = []
            epochConfusionMatrix = 0
            epochAccuracy = 0
            for batchNum, batches in enumerate(utils.get_minibatches(devData, config["batchSize"], False)):
                feedDict = {
                    comments: batches[0],
                    masks: batches[1],
                    labels: batches[5],
                    learningRate: config["learningRates"][epoch],
                    dropoutKeepProb: 1.0
                }
                if config["addCommentp"]:
                    feedDict[commentps] = batches[2]
                    feedDict[maskps] = batches[3]
                if config["addCommentf"]:
                    feedDict[commentfs] = batches[4]

                batchPredictions, batchConfusionMatrix, batchAccuracy = sess.run([prediction2, confusionMatrix, accuracy], feedDict)

                batchSize = len(batches[0])
                epochPredictions.extend(batchPredictions)
                epochConfusionMatrix += np.asarray(batchConfusionMatrix)
                epochAccuracy += batchAccuracy * batchSize
            devAccuracies.append(epochAccuracy / float(config["numDev"]))
            precision = epochConfusionMatrix[1][1] / float(epochConfusionMatrix[0][1] + epochConfusionMatrix[1][1])
            recall = epochConfusionMatrix[1][1] / float(epochConfusionMatrix[1][0] + epochConfusionMatrix[1][1])
            print "Dev Accuracy: {0:.4f} Precision: {1:.4f} Recall: {2:.4f} F1: {3:.4f}".format(
                devAccuracies[-1],
                precision,
                recall,
                2*precision*recall/(precision + recall))
            print epochConfusionMatrix

            # Save best dev 
            if bestDevIndex is None or devAccuracies[-1] > devAccuracies[bestDevIndex]:
                bestDevIndex = len(devAccuracies) - 1
                bestDevPredictions = epochPredictions
                bestDevConfusionMatrix = epochConfusionMatrix
                if config["saveDir"]:
                    savePath = saver.save(sess, config["saveDir"])
                    print "Variables saved at {}".format(savePath)
            print "Epoch Time: {0:.4f} s".format(time.time() - start)
            print ""

    # Print out summary.
    print "Best Dev of {} at epoch {}, train acc: {}, train loss: {}".format(
        devAccuracies[bestDevIndex],
        bestDevIndex + 1,
        trainAccuracies[bestDevIndex],
        losses[bestDevIndex])
    print bestDevConfusionMatrix

    # Return series.
    return losses, trainAccuracies, devAccuracies, bestDevPredictions

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LSTM model")
    parser.add_argument("-i", "--inDir", help="Input directory of train/dev/test", required=True)
    parser.add_argument("-o", "--outPrefix", help="Output prefix for plot", required=True)
    parser.add_argument("-s", "--saveDir", help="Directory to save to", required=False)
    parser.add_argument("-r", "--restoreDir", help="Directory to restore from", required=False)
    args = parser.parse_args()

    print "Loading config"
    config = getConfig()

    print "Loading embeddings"
    embed = np.loadtxt(args.inDir + "/embed.txt", dtype=np.float32)

    print "Loading Training Data"
    trainData = utils.loadComments(args.inDir + "/ProcessedTrain", config["numTrain"], config)

    print "Loading Dev Data"
    devData = utils.loadComments(args.inDir + "/ProcessedDev", config["numDev"], config)

    # Additional configs
    config["vocabSize"] = len(embed)
    config["embedDim"] = len(embed[0])
    config["numClasses"] = len(trainData[5][0])
    config["numCommentfs"] = len(trainData[4][0])
    utils.printConfig(config)
    config["saveDir"] = args.saveDir
    config["restoreDir"] = args.restoreDir
 
    print "Training"
    losses, trainAccuracies, devAccuracies, devPredictions = train(embed, trainData, devData, config)

    print "Plotting"
    utils.plot(losses, trainAccuracies, devAccuracies, args.outPrefix)

    print "Outputting"
    utils.labelCommentsWithPredictions(
        args.inDir + "/ProcessedDev",
        args.outPrefix + "Pred.json",
        devPredictions)
