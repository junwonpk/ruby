import os
import sys
import json
import time
import nltk
import pickle
import random
import numpy as np
import tensorflow as tf
from random import randrange
from tqdm import tqdm

nltk.download("punkt")

class ModelNetwork:
    def __init__(self, in_size, lstm_size, num_layers, out_size, session,
                 learning_rate=0.003, name="rnn"):
        self.scope = name
        self.in_size = in_size
        self.lstm_size = lstm_size
        self.num_layers = num_layers
        self.out_size = out_size
        self.session = session
        self.learning_rate = tf.constant(learning_rate)
        # Last state of LSTM, used when running the network in TEST mode
        self.lstm_last_state = np.zeros(
            (self.num_layers * 2 * self.lstm_size,)
        )
        with tf.variable_scope(self.scope):
            # (batch_size, timesteps, in_size)
            self.xinput = tf.placeholder(
                tf.float32,
                shape=(None, None, self.in_size),
                name="xinput"
            )
            self.lstm_init_value = tf.placeholder(
                tf.float32,
                shape=(None, self.num_layers * 2 * self.lstm_size),
                name="lstm_init_value"
            )
            # LSTM
            self.lstm_cells = [
                tf.contrib.rnn.BasicLSTMCell(
                    self.lstm_size,
                    forget_bias=1.0,
                    state_is_tuple=False
                ) for i in range(self.num_layers)
            ]
            self.lstm = tf.contrib.rnn.MultiRNNCell(
                self.lstm_cells,
                state_is_tuple=False
            )
            # Iteratively compute output of recurrent network
            outputs, self.lstm_new_state = tf.nn.dynamic_rnn(
                self.lstm,
                self.xinput,
                initial_state=self.lstm_init_value,
                dtype=tf.float32
            )
            # Linear activation (FC layer on top of the LSTM net)
            self.rnn_out_W = tf.Variable(
                tf.random_normal(
                    (self.lstm_size, self.out_size),
                    stddev=0.01
                )
            )
            self.rnn_out_B = tf.Variable(
                tf.random_normal(
                    (self.out_size,), stddev=0.01
                )
            )
            outputs_reshaped = tf.reshape(outputs, [-1, self.lstm_size])
            network_output = tf.matmul(
                outputs_reshaped,
                self.rnn_out_W
            ) + self.rnn_out_B
            batch_time_shape = tf.shape(outputs)
            self.final_outputs = tf.reshape(
                tf.nn.softmax(network_output),
                (batch_time_shape[0], batch_time_shape[1], self.out_size)
            )
            # Training: provide target outputs for supervised training.
            self.y_batch = tf.placeholder(
                tf.float32,
                (None, None, self.out_size)
            )
            y_batch_long = tf.reshape(self.y_batch, [-1, self.out_size])
            self.cost = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(
                    logits=network_output,
                    labels=y_batch_long
                )
            )
            self.train_op = tf.train.RMSPropOptimizer(
                self.learning_rate,
                0.9
            ).minimize(self.cost)

    # Input: X is a single element, not a list!
    def run_step(self, x, init_zero_state=True):
        # Reset the initial state of the network.
        if init_zero_state:
            init_value = np.zeros((self.num_layers * 2 * self.lstm_size,))
        else:
            init_value = self.lstm_last_state
        out, next_lstm_state = self.session.run(
            [self.final_outputs, self.lstm_new_state],
            feed_dict={
                self.xinput: [x],
                self.lstm_init_value: [init_value]
            }
        )
        self.lstm_last_state = next_lstm_state[0]
        return out[0][0]

    # xbatch must be (batch_size, timesteps, input_size)
    # ybatch must be (batch_size, timesteps, output_size)
    def train_batch(self, xbatch, ybatch):
        init_value = np.zeros(
            (xbatch.shape[0], self.num_layers * 2 * self.lstm_size)
        )
        cost, _ = self.session.run(
            [self.cost, self.train_op],
            feed_dict={
                self.xinput: xbatch,
                self.y_batch: ybatch,
                self.lstm_init_value: init_value
            }
        )
        return cost


def embed_to_vocab(data_, vocab):
    """
    Embed string to character-arrays -- it generates an array len(data)
    x len(vocab).
    Vocab is a list of elements.
    """
    data = np.zeros((len(data_), len(vocab)))
    cnt = 0
    for s in data_:
        v = [0.0] * len(vocab)
        v[vocab.index(s)] = 1.0
        data[cnt, :] = v
        cnt += 1
    return data


def decode_embed(array, vocab):
    return vocab[array.index(1)]

def compute_perplexity(loss):
    return tf.exp(loss)


def load_data(inputs):
    # Load the data
    data_ = []
    startwords = set()
    charlens = list()
    count = 0
    for i in tqdm(range(len(inputs))):
        input = inputs[i]
        # print("Loading {}".format(input))
        with open(input, 'r') as f:
            for i, line in enumerate(f, 1):
                count += 1
                raw = json.loads(line)
                comment = nltk.word_tokenize(raw["body"])
                data_ += comment
                if (len(comment)>0):
                    charlens.append(len(raw["body"]))
                    startwords.add(comment[0])
#                 data_ += comment["body"]
#                 charlens.append((len(comment["body"])))
#                 startwords.add(nltk.word_tokenize(comment["body"])[0])
                if count % 1000000 == 0:
                    print ("Processed {} lines".format(count))
    startwords = list(startwords)
    data_ = data_
    # Convert to 1-hot coding
    vocab = sorted(list(set(data_)))
    data = embed_to_vocab(data_, vocab)
    return data, vocab, startwords, charlens

def check_restore_parameters(sess, saver):
    """ Restore the previously trained parameters if there are any. """
    ckpt = tf.train.get_checkpoint_state(os.path.dirname('saved/checkpoint'))
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)

# def generate(input_files, config):
def generate(input_files, config, data, vocab, startwords, charlens, savefile):
    ckpt_file = "saved/model.ckpt"

#     print("Loading Data")
#     data, vocab, startwords, charlens = load_data(input_files)
    print("Loading Config")
    prefix = random.sample(startwords, 1)
    in_size = out_size = len(vocab)

    lstm_size = config["lstm_size"]
    learning_rate = config["lr_rate"]
    num_layers = config["num_layers"]
    batch_size = config["batch_size"]
    time_steps = config["time_steps"]
    NUM_TRAIN_BATCHES = config["NUM_TRAIN_BATCHES"]

    print("Config Loaded")
    # Number of test characters of text to generate after training the network
    LEN_TEST_TEXT = int(np.random.normal(np.mean(charlens),np.std(charlens)/2.0))

    # Initialize the network
    print("Initializaing Network")
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.InteractiveSession(config=config)
    net = ModelNetwork(
        in_size=in_size,
        lstm_size=lstm_size,
        learning_rate=learning_rate,
        num_layers=num_layers,
        out_size=out_size,
        session=sess,
        name="char_rnn_network"
    )
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(tf.global_variables())

    # 1. TRAIN THE NETWORK
    # check_restore_parameters(sess, saver)
    print("Trining Network")
    last_time = time.time()
    batch = np.zeros((batch_size, time_steps, in_size))
    batch_y = np.zeros((batch_size, time_steps, in_size))
    possible_batch_ids = range(data.shape[0] - time_steps - 1)

    for i in tqdm(range((NUM_TRAIN_BATCHES))):
        # Sample time_steps consecutive samples from the dataset text file
        batch_id = random.sample(possible_batch_ids, batch_size)

        for j in range(time_steps):
            ind1 = [k + j for k in batch_id]
            ind2 = [k + j + 1 for k in batch_id]

            batch[:, j, :] = data[ind1, :]
            batch_y[:, j, :] = data[ind2, :]

        cst = net.train_batch(batch, batch_y)

        if (i % 100) == 0:
            new_time = time.time()
            diff = new_time - last_time
            last_time = new_time
            with open('results/' + savefile + '.csv', 'a') as csvfile:
                csvwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                csvwriter.writerow([i, compute_perplexity(cst)])
            print(
                "batch: {}  loss: {}  pp: {} speed: {} batches / s".format(
                    i, cst, compute_perplexity(cst), 100 / diff
                )
            )
            saver.save(sess, ckpt_file)

    # 2) GENERATE LEN_TEST_TEXT CHARACTERS USING THE TRAINED NETWORK
    saver.restore(sess, ckpt_file)

    print("Generating Comment")
    TEST_PREFIX = prefix
#     TEST_PREFIX = TEST_PREFIX.lower()
    for i in range(len(TEST_PREFIX)):
        print(TEST_PREFIX[i])
        out = net.run_step(embed_to_vocab([TEST_PREFIX[i]], vocab), i == 0)

    print("Sentence:")
    gen_str = TEST_PREFIX[0]
    print(gen_str)
    for i in range(LEN_TEST_TEXT):
        element = np.random.choice(range(len(vocab)), p=out)
        gen_str += ' ' + vocab[element]
        out = net.run_step(embed_to_vocab([vocab[element]], vocab), False)
    with open('results/' + savefile + '.txt', 'r') as outFile:
        outFile.write(gen_str)

    print(gen_str)

def get_targets(subreddits, features):
    files = []
    # subreddits = ["announcements", "funny", "AskReddit", "todayilearned", "science", "worldnews",
    #          "pics", "IAmA", "gaming", "videos", "movies", "aww", "Music", "blog", "gifs",
    #          "news", "explainlikeimfive", "askscience", "EarthPorn", "books"]
    # features = ["nc20","nc30","sc1000","ct2000"]
    path = "target/"

    for s in subreddits:
        for f in features:
            for year in range(13):
                for month in range(12):
                    y = 2005 + year
                    y = 2016
                    m = 1 + month
                    m = 1
                    if m < 10:
                        m = "0" + str(m)
                    filename = path + str(y) + str(m) + "target" + s + f +".json"
                    if os.path.isfile(filename) and filename not in files:
                        files.append(filename)
    return files

if __name__ == "__main__":
    if sys.argv[1] == "test":
        config = {
            "lr_rate" : 0.003,
            "lstm_size" : 256,
            "num_layers" : 4,
            "batch_size" : 128,
            "time_steps" : 100,
            "NUM_TRAIN_BATCHES" : 100,
        }
        subreddits = ["funny"]
        features = ["nc30"]
        targets = get_targets(subreddits, features)

        print("Loading Data")
        data, vocab, startwords, charlens = load_data(targets)
        savefile = "{}-{}-{}-{}-{}".format(
            config["lr_rate"],
            config["num_layers"],
            config["batch_size"],
            config["NUM_TRAIN_BATCHES"],
            config["time_steps"],
        )
        generate(targets, config, data, vocab, startwords, charlens, savefile)
    elif sys.argv[1] == "batch":
        for n_batch in []:
            for b_size in []:
                for t_steps in []:
                    for n_layers in []:
                        for lr_rate in []:
                            for sub in []:
                                for fea in []:
                                    config = {
                                        "lr_rate" : lr_rate,
                                        "lstm_size" : 256,
                                        "num_layers" : n_layers,
                                        "batch_size" : b_size,
                                        "time_steps" : t_steps,
                                        "NUM_TRAIN_BATCHES" : n_batch,
                                    }
                                    targets = get_targets(sub, fea)
                                    print("Loading Data")
                                    data, vocab, startwords, charlens = load_data(targets)
                                    savefile = "{}-{}-{}-{}-{}-{}-{}".format(
                                        sub,
                                        fea,
                                        config["lr_rate"],
                                        config["num_layers"],
                                        config["batch_size"],
                                        config["NUM_TRAIN_BATCHES"],
                                        config["time_steps"],
                                    )
                                    generate(targets, config, data, vocab, startwords, charlens, savefile)
    else:
        print "Specify test or batch"
