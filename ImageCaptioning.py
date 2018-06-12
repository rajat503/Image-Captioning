import os
import json
import cv2
import string
import random
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

NUM_EPOCHS = 2000
BATCH_SIZE = 32
SEQUENCE_LENGTH = 12
LSTM_SIZE = 128

image_to_id = {}
id_to_caption = {}

with open('coco/annotations/captions_train2014.json') as f:
    train_captions = json.load(f)

for entry in train_captions["images"]:
    image_to_id[entry["file_name"]] = entry["id"]
for entry in train_captions["annotations"]:
    id_to_caption[entry["image_id"]] = entry["caption"]

vocab = set()
vocab_freq = {}
vocab_all = set()
def build_vocab():
    max_caption_len = 0
    for i in id_to_caption.keys():
        caption = id_to_caption[i]
        caption = str(caption.lower()).translate(None, string.punctuation)
        id_to_caption[i] = caption
        tokens = caption.split()
        max_caption_len = max(len(tokens), max_caption_len)
        for token in tokens:
            if token in vocab_all:
                vocab_freq[token] += 1
            else:
                vocab_freq[token] = 1
                vocab_all.add(token)
    for token in vocab_freq:
        if vocab_freq[token] >= 5:
            vocab.add(token)
    return max_caption_len

max_caption_len = min(build_vocab(),SEQUENCE_LENGTH)
print len(vocab)

GLOVE_PATH = "./glove.840B.300d.txt"
GLOVE_DIM = 300
glove_embeddings = []
word_to_index = {}
index_to_word = {}
count = 0
with open(GLOVE_PATH) as f:
    for line in f:
        entry = line.split(" ")
        word = entry[0].lower()
        if word in vocab and word not in word_to_index:
            glove_embeddings.append(list(map(float, entry[1:])))
            word_to_index[word] = count
            index_to_word[count] = word
            count += 1
word_to_index["<start_token>"] = count
index_to_word[count] = "<start_token>"
glove_embeddings.append([0.005] * GLOVE_DIM)
count += 1

word_to_index["<end_token>"] = count
index_to_word[count] = "<end_token>"
glove_embeddings.append([0.001] * GLOVE_DIM)
count += 1

word_to_index["<unk_token>"] = count
index_to_word[count] = "<unk_token>"
count += 1
glove_embeddings.append([0.0] * GLOVE_DIM)

vocab_size = len(glove_embeddings)

def encode_caption(caption, encoding_type, length=max_caption_len):
    count = 0
    encoding = []
    if encoding_type=="input":
        encoding.append([word_to_index["<start_token>"]])
        count += 1
    for token in caption.split():
        try:
            encoding.append([word_to_index[token]])
        except:
            encoding.append([word_to_index["<unk_token>"]])
        count += 1
        if count == max_caption_len or (count+1 == max_caption_len and encoding_type=="output"):
            break
    if encoding_type=="output":
        encoding.append([word_to_index["<end_token>"]])
        count += 1
    while count < max_caption_len:
        encoding.append([word_to_index["<end_token>"]])
        count += 1
    return encoding

def process_images(image_name):
    img = cv2.imread("./coco/train2014/"+image_name)
    img = cv2.resize(img, (224,224))
    return img

dataset = []
for image_name in image_to_id.keys():
    img_id = image_to_id[image_name]
    caption = id_to_caption[img_id]
    dataset.append((process_images(image_name),encode_caption(caption,"input"), encode_caption(caption,"output"), caption))

input_img = tf.placeholder(tf.float32, [BATCH_SIZE, 224, 224, 3])
input_seq = tf.placeholder(tf.int32, [SEQUENCE_LENGTH, BATCH_SIZE])
target_seq = tf.placeholder(tf.int32, [SEQUENCE_LENGTH, BATCH_SIZE])
lr = tf.placeholder(tf.float32)

preprocess_img = tf.keras.applications.resnet50.preprocess_input(input_img, mode="tf")
resnet = tf.keras.applications.ResNet50(include_top=False, weights='imagenet',input_tensor=input_img,input_shape=(224,224,3))
for layer in resnet.layers:
    layer.trainable = False
img_features = tf.reshape(resnet.outputs[0], [BATCH_SIZE, 2048])
img_embedding = tf.layers.dense(img_features, GLOVE_DIM)

lstm = tf.contrib.rnn.LSTMCell(LSTM_SIZE, initializer=tf.initializers.orthogonal(), activation=tf.nn.relu)
state = lstm.zero_state(BATCH_SIZE, tf.float32)
lstm_output, state = lstm(img_embedding, state)

embedding_matrix = tf.constant(glove_embeddings)
word_embeddings = tf.nn.embedding_lookup(embedding_matrix, input_seq)

input_sequence = tf.unstack(word_embeddings)
lstm_outputs = []

for input_batch in input_sequence:
    lstm_output, state = lstm(input_batch, state)
    lstm_outputs.append(lstm_output)

lstm_outputs = tf.stack(lstm_outputs)

fc = tf.layers.dense(lstm_outputs, 1024, activation=tf.nn.relu)
log_probs = tf.layers.dense(fc, vocab_size)
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target_seq,logits=log_probs))
update = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)
output = tf.argmax(log_probs, axis = 2)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for epoch in range(NUM_EPOCHS):
    index = 0
    num_examples = len(dataset)
    while index + BATCH_SIZE <=  num_examples:
        x_image = []
        y_input = []
        y_target = []
        mini_batch = dataset[index:index+BATCH_SIZE]
        for example in mini_batch:
            x_image.append(example[0])
            if len(y_input) == 0:
                y_input = np.array(example[1])
                y_target = np.array(example[2])
            else:
                y_input = np.hstack((y_input, example[1]))
                y_target = np.hstack((y_target, example[2]))
        x_image = np.array(x_image)
        index = index + BATCH_SIZE
        _, loss_, o = sess.run([update, loss,output], feed_dict={input_img: x_image,
                                                       input_seq: y_input,
                                                       target_seq: y_target,
                                                                lr: 0.0002})
        if index%10000 == 0:
            print "epoch=", epoch, "index=",index, "loss=", loss_

inp =  []
y_input = []
example = dataset[3368]
for i in range(BATCH_SIZE):
    inp.append(example[0])
    if len(y_input) == 0:
        y_input = np.array(example[1])
    else:
        y_input = np.hstack((y_input, example[1]))
inp = np.array(inp)
for i in range(SEQUENCE_LENGTH):
    pred = sess.run(output, feed_dict={input_img: np.array(inp), input_seq: y_input})
    print index_to_word[pred[i][0]]
    try:
        y_input[i+1][0] = pred[i][0]
    except:
        continue

print example[3]

plt.imshow(cv2.cvtColor(example[0], cv2.COLOR_BGR2RGB))
