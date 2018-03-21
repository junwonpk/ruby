import argparse
import numpy as np
import tensorflow as tf

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
        "addSIT": False,
        "addCommentp": False,
        "addDepth": False
    }
    config["addCommentf"] = config["addRT"] or config["addTime"] or config["addLength"] or config["addTime2"]
    config["learningRates"] = [0.01] * 2 + [0.005] * 3 + [0.003] * 5 + [0.002] * 3 + [0.001] * 2
    config["lstmUnits"] = 128
    config["attentionUnits"] = 32
    config["layer2Units"] = 32
    config["numDev"] = 40000
    config["numEpochs"] = len(config["learningRates"])
    config["predictScore"] = False

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

    with tf.variable_scope('v'):
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
    with tf.variable_scope(scope):
        # LSTM
        seqLengths = tf.reduce_sum(masks, axis=1)
        lstmCell = tf.contrib.rnn.BasicLSTMCell(config["lstmUnits"])
        # lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=dropoutKeepProb)
        cellOutputs, _ = tf.nn.dynamic_rnn(lstmCell, embeddings, sequence_length=seqLengths, dtype=tf.float32, scope=scope)

        # Attention layer
        attentionOutputs, alphas = attention(cellOutputs, config["attentionUnits"], False, True)

        # Dropout layer
        dropoutOutputs = tf.nn.dropout(attentionOutputs, dropoutKeepProb)

        return dropoutOutputs, alphas

def getLSTMOutputs(embeddings, masks, dropoutKeepProb, scope, config):
    with tf.variable_scope(scope):
        # LSTM
        lstmCell = tf.contrib.rnn.BasicLSTMCell(config["lstmUnits"])
        # lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=dropoutKeepProb)
        cellOutputs, _ = tf.nn.dynamic_rnn(lstmCell, embeddings, dtype=tf.float32, scope=scope)

        # Output to pred
        cellOutputs = tf.transpose(cellOutputs, [2, 0, 1]) # cells, batches, len
        maskedOutputs = tf.reduce_sum(cellOutputs * masks, axis=2) / tf.reduce_sum(masks, axis=1)
        lstmOutputs = tf.transpose(maskedOutputs, [1, 0]) # batches, cells

    return lstmOutputs

def getAlphas(embed, devData, config):
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
        embeddings = tf.nn.embedding_lookup(E, comments)
        embeddingps = tf.nn.embedding_lookup(E, commentps)

    # LSTM
    lstmOutputs, alphas = getAttentionLSTMOutputs(embeddings, masks, dropoutKeepProb, "lstm", config)
    lstmOutputs = [lstmOutputs]
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

    # Dropout
    layer1Droutput = tf.nn.dropout(layer1Output, dropoutKeepProb)

    # layer 2 softmax
    with tf.variable_scope("layer2"):
        W2 = tf.get_variable(
            "W2",
            shape=[config["layer2Units"], config["numClasses"]],
            initializer=tf.initializers.truncated_normal())
        b2 = tf.get_variable(
            "b2",
            shape=[config["numClasses"]],
            initializer=tf.constant_initializer(0.1))
    prediction = tf.matmul(layer1Droutput, W2) + b2

    # Accuracy
    prediction2 = tf.argmax(prediction, 1)

    # Saver
    saver = tf.train.Saver()

    with tf.Session() as sess:
        # Variable Initialization.
        saver.restore(sess, config["restoreDir"])

        # Evaluate on dev and get attention weights.
        predictions = []
        allAlphas = None
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

            batchAlphas, = sess.run([alphas], feedDict)
            if allAlphas is None:
                allAlphas = batchAlphas
            else:
                allAlphas = np.concatenate((allAlphas, batchAlphas), axis=0)

            if (batchNum + 1) % 100 == 0:
                print "Batch: {}".format(batchNum + 1)

    # Return series.
    return allAlphas

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LSTM model")
    parser.add_argument("-i", "--inDir", help="Input directory of train/dev/test", required=True)
    parser.add_argument("-o", "--outFile", help="Prediction output filename", required=True)
    parser.add_argument("-r", "--restoreDir", help="Directory to restore from", required=False)
    args = parser.parse_args()

    print "Loading config"
    config = getConfig()

    print "Loading embeddings"
    embed = np.loadtxt(args.inDir + "/embed.txt", dtype=np.float32)

    print "Loading Dev Data"
    devData = utils.loadComments(args.inDir + "/ProcessedDev", config["numDev"], config)

    # Additional configs
    config["vocabSize"] = len(embed)
    config["embedDim"] = len(embed[0])
    config["numClasses"] = len(devData[5][0])
    config["numCommentfs"] = len(devData[4][0])
    config["numLSTMOutputs"] = config["lstmUnits"] + config["numCommentfs"]
    if config["addCommentp"]:
        config["numLSTMOutputs"] += config["lstmUnits"]
    utils.printConfig(config)
    config["restoreDir"] = args.restoreDir
 
    print "Fetching"
    alphas = getAlphas(embed, devData, config)

    print "Outputting"
    np.savetxt(args.outFile, alphas)

