import argparse
import numpy as np
import tensorflow as tf

import utils

# IMPORTANT: Contains the configurations for the LSTM
def getConfig():
    config = {
        "maxDocLength": 250,  # Max is 2191
        "batchSize": 256,
        "addRT": True,
        "addTime": False,
        "addTime2": False,
        "addLength": True,
        "addCommentp": False
    }
    config["addCommentf"] = config["addRT"] or config["addTime"] or config["addLength"]
    config["learningRates"] = [0.01] * 5 + [0.005] * 5 + [0.003] * 5 + [0.002] * 5
    config["lstmUnits"] = 64
    config["attentionUnits"] = None
    config["layer2Units"] = 32
    config["numClasses"] = 2
    config["dropoutKeepProb"] = 0.9
    config["numTrain"] = 200000
    config["numDev"] = 40000
    config["numEpochs"] = len(config["learningRates"])

    # Junk.
    # config["learningRates"] = [0.01] * 5 + [0.005] * 5 + [0.003] * 5 + [0.002] * 5 + [0.001] * 5 + [0.0005] * 5 + [0.0003] * 5 + [0.0001] * 5
    # config["learningRates"] = [0.01] * 3 + [0.005] * 2 + [0.003] * 5 + [0.002] * 10 + [0.001] * 5 + [0.0005] * 5 + [0.0003] * 5 + [0.0001] * 5
    # config["learningRates"] = [0.01] * 3 + [0.005] * 2 + [0.003] * 5 + [0.002] * 10 + [0.001] * 5 + [0.0005] * 5
    # learningRates = [0.01] * 10 + [0.005] * 10 + [0.003] * 10 + [0.002] * 10
    # learningRates = [0.01] * 3 + [0.005] * 2 + [0.003] * 5 + [0.002] * 10 + [0.001] * 5 + [0.0005] * 5 + [0.0004] * 5 + [0.0003] * 5 + [0.0002] * 5 + [0.0001] * 5
    # learningRates = [0.01] * 10 + [0.005] * 10

    return config

def attention(inputs, attention_size, time_major=False, return_alphas=False):
    if isinstance(inputs, tuple):
        # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
        inputs = tf.concat(inputs, 2)

    if time_major:
        # (T,B,D) => (B,T,D)
        inputs = tf.array_ops.transpose(inputs, [1, 0, 2])

    hidden_size = inputs.shape[2].value  # D value - hidden size of the RNN layer

    # Trainable parameters
    w_omega = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
    b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
    u_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))

    with tf.name_scope('v'):
        # Applying fully connected layer with non-linear activation to each of the B*T timestamps;
        #  the shape of `v` is (B,T,D)*(D,A)=(B,T,A), where A=attention_size
        v = tf.tanh(tf.tensordot(inputs, w_omega, axes=1) + b_omega)

    # For each of the timestamps its vector of size A from `v` is reduced with `u` vector
    vu = tf.tensordot(v, u_omega, axes=1, name='vu')  # (B,T) shape
    alphas = tf.nn.softmax(vu, name='alphas')         # (B,T) shape

    # Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape
    output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)

    if not return_alphas:
        return output
    else:
        return output, alphas

def getAttentionLSTMOutputs(embeddings, masks, dropoutKeepProb, scope, config):
    with tf.name_scope(scope):
        # LSTM
        seqLengths = tf.reduce_sum(masks, axis=1)
        lstmCell = tf.contrib.rnn.BasicLSTMCell(config["lstmUnits"])
        lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=dropoutKeepProb)
        cellOutputs, _ = tf.nn.dynamic_rnn(lstmCell, embeddings, sequence_length=seqLengths, dtype=tf.float32, scope=scope)

        # Attention layer
        attentionOutputs = attention(cellOutputs, config["attentionUnits"])

        # Dropout layer
        dropoutOutputs = tf.nn.dropout(attentionOutputs, dropoutKeepProb)

        return dropoutOutputs

def getLSTMOutputs(embeddings, masks, dropoutKeepProb, scope, config):
    with tf.name_scope(scope):
        # LSTM
        lstmCell = tf.contrib.rnn.BasicLSTMCell(config["lstmUnits"])
        lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=dropoutKeepProb)
        cellOutputs, _ = tf.nn.dynamic_rnn(lstmCell, embeddings, dtype=tf.float32, scope=scope)

        # Output to pred
        cellOutputs = tf.transpose(cellOutputs, [2, 0, 1]) # cells, batches, len
        maskedOutputs = tf.reduce_sum(cellOutputs * masks, axis=2) / tf.reduce_sum(masks, axis=1)
        lstmOutputs = tf.transpose(maskedOutputs, [1, 0]) # batches, cells

    return lstmOutputs

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
    with tf.name_scope("embedding"):
        E = tf.get_variable("E", initializer=embed, trainable=False)
        embeddings = tf.nn.embedding_lookup(E, comments)
        embeddingps = tf.nn.embedding_lookup(E, commentps)

    # LSTM
    lstmOutputs = None
    if config["attentionUnits"]:
        lstmOutputs = [getAttentionLSTMOutputs(embeddings, masks, dropoutKeepProb, "lstm", config)]
    else:
        lstmOutputs = [getLSTMOutputs(embeddings, masks, dropoutKeepProb, "lstm", config)]
    if config["addCommentp"]:
        lstmOutputs.append(getLSTMOutputs(embeddingps, maskps, dropoutKeepProb, "lstmp", config))
    if config["addCommentf"]:
        lstmOutputs.append(commentfs)
    lstmOutputs = tf.concat(lstmOutputs, axis=1)

    # Layer 1 ReLu
    W1 = tf.get_variable(
        "W1",
        shape=[config["numLSTMOutputs"], config["layer2Units"]],
        initializer=tf.initializers.truncated_normal())
    b1 = tf.get_variable(
        "b1", 
        shape=[config["layer2Units"]], 
        initializer=tf.constant_initializer(0.1))
    layer1Output = tf.nn.relu(tf.matmul(lstmOutputs, W1) + b1)

    # layer 2 softmax
    with tf.name_scope("layer2"):
        W2 = tf.get_variable(
            "W2",
            shape=[config["layer2Units"], config["numClasses"]],
            initializer=tf.initializers.truncated_normal())
        b2 = tf.get_variable(
            "b2",
            shape=[config["numClasses"]],
            initializer=tf.constant_initializer(0.1))
    prediction = tf.matmul(layer1Output, W2) + b2

    # Accuracy
    prediction2 = tf.argmax(prediction, 1)
    labels2 = tf.argmax(labels, 1)
    confusionMatrix = tf.confusion_matrix(labels2, prediction2)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction2, labels2), tf.float32))

    # Loss and optimizer
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels))
    optimizer = tf.train.AdamOptimizer(learning_rate=learningRate).minimize(loss)

    # Saver
    # saver = tf.train.Saver()

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
        # if saveIn:
        #     saver.restore(sess, saveIn)
        # else:
        sess.run(tf.global_variables_initializer())

        for epoch in range(config["numEpochs"]):
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
            print "Dev Accuracy: {}".format(devAccuracies[-1])

            # Save best dev 
            if bestDevIndex is None or devAccuracies[-1] > devAccuracies[bestDevIndex]:
                bestDevIndex = len(devAccuracies) - 1
                bestDevPredictions = epochPredictions
                bestDevConfusionMatrix = epochConfusionMatrix

            # savePath = saver.save(sess, saveOut)
            # print "Model saved at {}".format(savePath)

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
    args = parser.parse_args()

    print "Loading config"
    config = getConfig()

    print "Loading embeddings"
    embed = np.loadtxt(args.inDir + "/embed.txt", dtype=np.float32)

    print "Loading Training Data"
    trainData = utils.loadComments(args.inDir + "/ProcessedDev", config["numTrain"], config)

    print "Loading Dev Data"
    devData = utils.loadComments(args.inDir + "/ProcessedTrain", config["numDev"], config)

    # Additional configs
    config["vocabSize"] = len(embed)
    config["embedDim"] = len(embed[0])
    config["numCommentfs"] = len(trainData[4][0])
    config["numLSTMOutputs"] = config["lstmUnits"] + config["numCommentfs"]
    if config["addCommentp"]:
        config["numLSTMOutputs"] += config["lstmUnits"]
    utils.printConfig(config)
 
    print "Training"
    losses, trainAccuracies, devAccuracies, devPredictions = train(embed, trainData, devData, config)

    print "Plotting"
    utils.plot(losses, trainAccuracies, devAccuracies, args.outPrefix)

    print "Outputting"
    utils.labelCommentsWithPredictions(
        args.inDir + "/Reddit2ndDevT",
        args.outPrefix + "Pred.json",
        devPredictions)
