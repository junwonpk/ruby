import argparse
import numpy as np
import tensorflow as tf
import json

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
    config["dropoutKeepProb"] = 0.9
    config["predictScore"] = False
    config["filterSizes"] = [1, 2, 3, 4]
    config["numFilters"] = 20
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

def evaluate(embed, devData, config):
    # Create input placeholders
    comments = tf.placeholder(tf.int32, [None, config["maxDocLength"]])
    masks = tf.placeholder(tf.float32, [None, config["maxDocLength"]])
    commentps = tf.placeholder(tf.int32, [None, config["maxDocLength"]])
    maskps = tf.placeholder(tf.float32, [None, config["maxDocLength"]])
    commentfs = tf.placeholder(tf.float32, [None, config["numCommentfs"]])
    labels = tf.placeholder(tf.float32, [None, config["numClasses"]])
    dropoutKeepProb = tf.placeholder(tf.float32)

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
    soft = tf.nn.softmax(prediction)

    # Accuracy
    prediction2 = tf.argmax(prediction, 1)
    labels2 = tf.argmax(labels, 1)
    confusionMatrix = tf.confusion_matrix(labels2, prediction2)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction2, labels2), tf.float32))

    # Saver
    saver = tf.train.Saver()

    with tf.Session() as sess:
        # Variable Initialization.
        if config["restoreDir"]:
            print "Restoring from {}".format(config["restoreDir"])
            saver.restore(sess, config["restoreDir"])
        else:
            sess.run(tf.global_variables_initializer())

        # Evaluate on test
        epochPredictions = []
        epochAccuracy = 0
        epochConfusionMatrix = 0
        totalNum = 0
        for batchNum, batches in enumerate(utils.get_minibatches(devData, config["batchSize"], False)):
            feedDict = {
                comments: batches[0],
                masks: batches[1],
                labels: batches[5],
                dropoutKeepProb: 1.0
            }
            if config["addCommentp"]:
                feedDict[commentps] = batches[2]
                feedDict[maskps] = batches[3]
            if config["addCommentf"]:
                feedDict[commentfs] = batches[4]

            batchSize = len(batches[0])
            batchPredictions, batchConfusionMatrix, batchAccuracy = sess.run([soft, confusionMatrix, accuracy], feedDict)

            epochPredictions.extend([pred.tolist() for pred in batchPredictions])
            epochConfusionMatrix += np.asarray(batchConfusionMatrix)
            epochAccuracy += batchAccuracy * batchSize
            totalNum += batchSize

            if (batchNum + 1) % 100 == 0:
                print "Batch: {}".format(batchNum + 1)
        precision = epochConfusionMatrix[1][1] / float(epochConfusionMatrix[0][1] + epochConfusionMatrix[1][1])
        recall = epochConfusionMatrix[1][1] / float(epochConfusionMatrix[1][0] + epochConfusionMatrix[1][1])
        print "Test Accuracy: {0:.4f} Precision: {1:.4f} Recall: {2:.4f} F1: {3:.4f}".format(
            epochAccuracy / float(totalNum),
            precision,
            recall,
            2*precision*recall/(precision + recall))
        print epochConfusionMatrix

    # Return series.
    return epochPredictions

def labelCommentsWithSoft(inFilename, outFilename, predictions):
    with open(inFilename, "r") as inFile, open(outFilename, "w") as outFile:
        for i, line in enumerate(inFile):
            comment = json.loads(line)
            comment["soft"] = predictions[i]
            outFile.write(json.dumps(comment) + "\n")

            if (i + 1) % 10000 == 0:
                print "Processed {} lines".format(i + 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LSTM model")
    parser.add_argument("-i", "--inDir", help="Input directory of train/dev/test", required=True)
    parser.add_argument("-r", "--restoreDir", help="Directory to restore from", required=False)
    args = parser.parse_args()

    print "Loading config"
    config = getConfig()

    print "Loading embeddings"
    embed = np.loadtxt(args.inDir + "/embed.txt", dtype=np.float32)

    print "Loading Dev Data"
    devData = utils.loadComments(args.inDir + "/ProcessedDev", float('inf'), config)

    # Additional configs
    config["vocabSize"] = len(embed)
    config["embedDim"] = len(embed[0])
    config["numClasses"] = len(devData[5][0])
    config["numCommentfs"] = len(devData[4][0])
    utils.printConfig(config)
    config["saveDir"] = None
    config["restoreDir"] = args.restoreDir
 
    print "Evaluating"
    devPredictions = evaluate(embed, devData, config)

    labelCommentsWithSoft(
        "../statsout/T49/summaryPred.json",
        "../statsout/T49/summaryPred2.json",
        devPredictions)


