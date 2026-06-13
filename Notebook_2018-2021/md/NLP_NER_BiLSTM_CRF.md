---
jupyter:
  jupytext:
    notebook_metadata_filter: -jupytext.text_representation.jupytext_version
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

```python id="sKmnHpUghSMr" outputId="22ef37b4-064b-45c7-d1cc-ff93f77ec4c7"
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
tf.__version__
```

```python id="P4Llk5YxhSMs" outputId="f000e159-4c12-42d0-8d0f-195a4407c507"
tfds.__version__
```

<!-- #region id="mKv9q7ZohSMt" -->

# Only if you have a GPU
<!-- #endregion -->

```python id="8OC3EJHBhSMu" outputId="26b18b89-bec7-4d1d-ce2e-501fe11dac66"
######## GPU CONFIGS FOR RTX 2070 ###############
## Please ignore if not training on GPU       ##
## this is important for running CuDNN on GPU ##

tf.keras.backend.clear_session() #- for easy reset of notebook state

# chck if GPU can be seen by TF
tf.config.list_physical_devices('GPU')
# only if you want to see how commands are executed, uncomment below
# tf.debugging.set_log_device_placement(True)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only use the first GPU
  try:
    tf.config.experimental.set_memory_growth(gpus[0], True)
    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
  except RuntimeError as e:
    # Visible devices must be set before GPUs have been initialized
    print(e)
###############################################

```

<!-- #region heading_collapsed=true id="wQz2WhRRhSMv" -->
# Loading the Data
<!-- #endregion -->

```python hidden=true id="cD_VWuibhSMv" outputId="a73c2fe1-0ca1-4b4d-8b78-c9b308830525"
!wget https://gmb.let.rug.nl/releases/gmb-2.2.0.zip
```

```python hidden=true id="TA8MEcbEhSMw" outputId="ec105bf1-8278-464b-ab32-47d1bb2b50f3"
# !unzip -o gmb-2.2.0.zip  <= use the -o to expand and overwrite whtout prompting
!unzip gmb-2.2.0.zip
```

```python hidden=true id="neUCAp0fhSMx"
import os
```

```python hidden=true id="2WrN3dZxhSMx"
data_root = './gmb-2.2.0/data/'

fnames = []
for root, dirs, files in os.walk(data_root):
    for filename in files:
        if filename.endswith(".tags"):
            fnames.append(os.path.join(root, filename))
```

```python hidden=true id="1n4RYW50hSMy" outputId="0423025e-c02f-4b7e-d8e5-8521962314fb"
fnames[:2]
```

```python hidden=true id="IpGGZKyWhSMz"
!mkdir ner
```

```python hidden=true id="naKb9MguhSM0"
import csv
import collections

ner_tags = collections.Counter()
iob_tags = collections.Counter()

def strip_ner_subcat(tag):
    # NER tags are of form {cat}-{subcat}
    # eg tim-dow. We only want first part
    return tag.split("-")[0]


def iob_format(ners):
    # converts IO tags into BIO format
    # input is a sequence of IO NER tokens
    # convert this: O, PERSON, PERSON, O, O, LOCATION, O
    # into: O, B-PERSON, I-PERSON, O, O, B-LOCATION, O
    iob_tokens = []
    for idx, token in enumerate(ners):
        if token != 'O':  # !other
            if idx == 0:
                token = "B-" + token #start of sentence
            elif ners[idx-1] == token:
                token = "I-" + token  # continues
            else:
                token = "B-" + token
        iob_tokens.append(token)
        iob_tags[token] += 1
    return iob_tokens

total_sentences = 0
outfiles = []
for idx, file in enumerate(fnames):
    with open(file, 'rb') as content:
        data = content.read().decode('utf-8').strip()
        sentences = data.split("\n\n")
        print(idx, file, len(sentences))
        total_sentences += len(sentences)

        with open("./ner/"+str(idx)+"-"+os.path.basename(file), 'w') as outfile:
            outfiles.append("./ner/"+str(idx)+"-"+os.path.basename(file))
            writer = csv.writer(outfile)

            for sentence in sentences:
                toks = sentence.split('\n')
                words, pos, ner = [], [], []

                for tok in toks:
                    t = tok.split("\t")
                    words.append(t[0])
                    pos.append(t[1])
                    ner_tags[t[3]] += 1
                    ner.append(strip_ner_subcat(t[3]))
                writer.writerow([" ".join(words),
                                 " ".join(iob_format(ner)),
                                 " ".join(pos)])
```

```python hidden=true id="zvZC3sAZhSM0" outputId="79c4d330-37fb-4338-c4ac-fdccaa2766cc"
print("total number of sentences: ", total_sentences)
```

```python hidden=true id="d8z5Z1tPhSM0" outputId="a5440862-75e0-483c-f305-fe821f2967a6"
print(ner_tags)
print(iob_tags)
```

```python hidden=true id="s9aFwQSEhSM1"
import matplotlib.pyplot as plt

labels, values = zip(*iob_tags.items())
```

```python hidden=true id="E_o6Azm7hSM1" outputId="1c81ae07-a9a9-4e27-d750-1af9b937d3a4"
indexes = np.arange(len(labels))


plt.bar(indexes, values)
plt.xticks(indexes, labels, rotation='vertical')
plt.margins(0.01)
plt.subplots_adjust(bottom=0.15)
plt.show()
```

<!-- #region id="7z0qCJzXhSM1" -->
# Normalizing and Vectorizing
<!-- #endregion -->

```python id="f0eRB2KghSM2"
import glob
import pandas as pd

# could use `outfiles` param as well
files = glob.glob("./ner/*.tags")

data_pd = pd.concat([pd.read_csv(f, header=None,
                                 names=["text", "label", "pos"])
                for f in files], ignore_index = True)
```

```python id="T-ywLg4vhSM2" outputId="6d1e0cc0-f690-4b37-9a67-e5ee30ffe85f"
data_pd.info()
```

```python id="_of4hKR5hSM3"
### Keras tokenizer
from tensorflow.keras.preprocessing.text import Tokenizer
text_tok = Tokenizer(filters='[\\]^\t\n', lower=False,
                     split=' ', oov_token='<OOV>')

pos_tok = Tokenizer(filters='\t\n', lower=False,
                    split=' ', oov_token='<OOV>')

ner_tok = Tokenizer(filters='\t\n', lower=False,
                    split=' ', oov_token='<OOV>')
```

```python id="ZAAWlT_5hSM3"
text_tok.fit_on_texts(data_pd['text'])
pos_tok.fit_on_texts(data_pd['pos'])
ner_tok.fit_on_texts(data_pd['label'])
```

```python id="Tq_A-isphSM3"
ner_config = ner_tok.get_config()
text_config = text_tok.get_config()
```

```python id="NXopKPRMhSM3" outputId="10863fcf-7f0b-43aa-a427-a53035584810"
print(ner_config)
```

```python id="_IJwps2NhSM3" outputId="8b88cb55-0d37-4054-e4f6-706e155cd9c0"
text_vocab = eval(text_config['index_word'])
ner_vocab = eval(ner_config['index_word'])

print("Unique words in vocab:", len(text_vocab))
print("Unique NER tags in vocab:", len(ner_vocab))
```

```python id="NhaCfgFZhSM4"
x_tok = text_tok.texts_to_sequences(data_pd['text'])
y_tok = ner_tok.texts_to_sequences(data_pd['label'])
```

```python id="AuB-0AuHhSM4" outputId="9f97f35b-f36a-4cf1-9bc2-dd6d08118e3f"
print(text_tok.sequences_to_texts([x_tok[1]]), data_pd['text'][1])
print(ner_tok.sequences_to_texts([y_tok[1]]), data_pd['label'][1])
```

```python id="D7f-dZeGhSM5"
# now, pad seqences to a maximum length
from tensorflow.keras.preprocessing import sequence

max_len = 50

x_pad = sequence.pad_sequences(x_tok, padding='post',
                              maxlen=max_len)
y_pad = sequence.pad_sequences(y_tok, padding='post',
                              maxlen=max_len)
```

```python id="kkA40eN5hSM5" outputId="4257d0bc-f7c2-4b51-8b14-c1561ee15423"
print(x_pad.shape, y_pad.shape)
```

```python id="qKJ3Of3WhSM5" outputId="af8e7e03-22f4-4f01-ce9d-99e4bf261d17"
text_tok.sequences_to_texts([x_pad[1]])
```

```python id="bV7dFhQOhSM5" outputId="b9f89378-6707-49ee-900f-d7dc1f0fa240"
ner_tok.sequences_to_texts([y_pad[1]])
```

```python id="UZD8uyKihSM5" outputId="a09aecbf-ea4b-4b32-bf75-24043f2160c8"
num_classes = len(ner_vocab)+1

Y = tf.keras.utils.to_categorical(y_pad, num_classes=num_classes)
Y.shape
```

<!-- #region id="oFWBnm7OhSM6" -->
# Building and Training the BiLSTM Model
<!-- #endregion -->

```python id="uErB3c-0hSM6"
# Length of the vocabulary
vocab_size = len(text_vocab) + 1

# The embedding dimension
embedding_dim = 64

# Number of RNN units
rnn_units = 100

#batch size
BATCH_SIZE=90

# num of NER classes
num_classes = len(ner_vocab)+1

from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, TimeDistributed, Dense

dropout=0.2
def build_model_bilstm(vocab_size, embedding_dim, rnn_units, batch_size, classes):
  model = tf.keras.Sequential([
    Embedding(vocab_size, embedding_dim, mask_zero=True,
                              batch_input_shape=[batch_size, None]),
    Bidirectional(LSTM(units=rnn_units,
                           return_sequences=True,
                           dropout=dropout,
                           kernel_initializer=tf.keras.initializers.he_normal())),
    TimeDistributed(Dense(rnn_units, activation='relu')),
    Dense(num_classes, activation="softmax")
  ])


  return model
```

```python id="YZeMjsFAhSM6" outputId="b224ad4b-11c0-4899-9959-1bfeca1b6d08"
model = build_model_bilstm(
                        vocab_size = vocab_size,
                        embedding_dim=embedding_dim,
                        rnn_units=rnn_units,
                        batch_size=BATCH_SIZE,
                        classes=num_classes)
model.summary()
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
```

```python id="tQXzCdtvhSM7"
X = x_pad
```

```python id="JFimHEB9hSM7"
# create training and testing splits
total_sentences = 62010
test_size = round(total_sentences / BATCH_SIZE * 0.2)
X_train = X[BATCH_SIZE*test_size:]
Y_train = Y[BATCH_SIZE*test_size:]

X_test = X[0:BATCH_SIZE*test_size]
Y_test = Y[0:BATCH_SIZE*test_size]
```

```python id="5XW9fJGthSM7" outputId="a3e0e419-dc54-4106-db19-675002158484"
print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)
```

```python id="rGJEH_PWhSM7" outputId="d839ba86-b8aa-43f9-ef43-1cb0f24fe7fe"
model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=15)
```

```python id="OlNMcxV8hSM8" outputId="5c1ee170-a7c3-4bcc-91ca-67160f992377"
# batch size in eval
model.evaluate(X_test, Y_test, batch_size=BATCH_SIZE)
```

```python id="jK_ZNX-YhSM8"
y_pred = model.predict(X_test, batch_size=BATCH_SIZE)
```

```python id="lCgdbclzhSM8" outputId="2fd6dfda-c7b7-4cf5-f759-6a26fc997d8c"
text_tok.sequences_to_texts([X_test[1]])
```

```python id="feEBUh3phSM8" outputId="c9e8e7a0-6d9f-4f34-a6f2-a0eec4e91a1d"
ner_tok.sequences_to_texts([y_pad[1]])
```

```python id="M7fXDvoChSM9" outputId="c138a2f6-6a95-4b74-d15e-21061cae2389"
y_pred = tf.argmax(y_pred, -1)
y_pred.shape
```

```python id="__leZilphSM9"
y_pnp = y_pred.numpy()
```

```python id="FzgNTfKphSM9" outputId="1ee144cd-307f-43ce-8900-f36b2f0e2ea8"
ner_tok.sequences_to_texts([y_pnp[1]])
```

<!-- #region id="biHbnzJohSM9" -->
## BiLSTM-CRF Model
<!-- #endregion -->

```python id="aWpjF21VhSM9" outputId="62cb29af-6daa-43cd-e9b9-2b75ae334f2b"
!pip install tensorflow_addons
```

```python id="hDs-OxMZhSM-" outputId="8d9ac6ed-a6ba-4869-e50c-1016c6b197c5"
import tensorflow_addons as tfa
tfa.__version__
```

```python id="EOIbPXUnhSM-"
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K

class CRFLayer(Layer):
  """
  Computes the log likelihood during training
  Performs Viterbi decoding during prediction
  """
  def __init__(self,
               label_size,
               mask_id=0,
               trans_params=None,
               name='crf',
               **kwargs):
    super(CRFLayer, self).__init__(name=name, **kwargs)
    self.label_size = label_size
    self.mask_id = mask_id
    self.transition_params = None

    if trans_params is None:  # not reloading pretrained params
        self.transition_params = tf.Variable(tf.random.uniform(shape=(label_size, label_size)),
                                         trainable=False)
    else:
        self.transition_params = trans_params

  def get_seq_lengths(self, matrix):
    # matrix is of shape (batch_size, max_seq_len)
    mask = tf.not_equal(matrix, self.mask_id)
    seq_lengths = tf.math.reduce_sum(
                                    tf.cast(mask, dtype=tf.int32),
                                    axis=-1)
    return seq_lengths

  def call(self, inputs, seq_lengths, training=None):
    if training is None:
        training = K.learning_phase()

    # during training, this layer just returns the logits
    if training:
        return inputs

    # viterbi decode logic to return proper
    # results at inference
    _, max_seq_len, _ = inputs.shape
    seqlens = seq_lengths
    paths = []
    for logit, text_len in zip(inputs, seqlens):
        viterbi_path, _ = tfa.text.viterbi_decode(logit[:text_len],
                                              self.transition_params)
        paths.append(self.pad_viterbi(viterbi_path, max_seq_len))

    return tf.convert_to_tensor(paths)

  def pad_viterbi(self, viterbi, max_seq_len):
    if len(viterbi) < max_seq_len:
        viterbi = viterbi + [self.mask_id] * (max_seq_len - len(viterbi))
    return viterbi

  def get_proper_labels(self, y_true):
    shape = y_true.shape
    if len(shape) > 2:
        return tf.argmax(y_true, -1, output_type=tf.int32)
    return y_true

  def loss(self, y_true, y_pred):
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(self.get_proper_labels(y_true), y_pred.dtype)

    seq_lengths = self.get_seq_lengths(y_true)
    log_likelihoods, self.transition_params = tfa.text.crf_log_likelihood(y_pred,
                                                                y_true, seq_lengths)

    # save transition params
    self.transition_params = tf.Variable(self.transition_params, trainable=False)
    # calc loss
    loss = - tf.reduce_mean(log_likelihoods)
    return loss

```

```python id="CAp7XmtuhSM_"
from tensorflow.keras import Model, Input, Sequential
from tensorflow.keras.layers import LSTM, Embedding, Dense, TimeDistributed
from tensorflow.keras.layers import Dropout, Bidirectional
from tensorflow.keras import backend as K

class NerModel(tf.keras.Model):
    def __init__(self, hidden_num, vocab_size, label_size, embedding_size,
                name='BilstmCrfModel', **kwargs):
        super(NerModel, self).__init__(name=name, **kwargs)
        self.num_hidden = hidden_num
        self.vocab_size = vocab_size
        self.label_size = label_size

        self.embedding = Embedding(vocab_size, embedding_size,
                                   mask_zero=True, name="embedding")
        self.biLSTM =Bidirectional(LSTM(hidden_num, return_sequences=True), name="bilstm")
        self.dense = TimeDistributed(tf.keras.layers.Dense(label_size), name="dense")
        self.crf = CRFLayer(self.label_size, name="crf")

    def call(self, text, labels=None, training=None):
        seq_lengths = tf.math.reduce_sum(tf.cast(tf.math.not_equal(text, 0),
                                               dtype=tf.int32), axis=-1)

        if training is None:
            training = K.learning_phase()

        inputs = self.embedding(text)
        bilstm = self.biLSTM(inputs)
        logits = self.dense(bilstm)
        outputs = self.crf(logits, seq_lengths, training)

        return outputs
```

```python id="gZD4uN1VhSM_"
# Length of the vocabulary in chars
vocab_size = len(text_vocab)+1 # len(chars)

# The embedding dimension
embedding_dim = 64

# Number of RNN units
rnn_units = 100

#batch size
BATCH_SIZE=90

# num of NER classes
num_classes = len(ner_vocab)+1

blc_model = NerModel(rnn_units, vocab_size, num_classes, embedding_dim, dynamic=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
```

```python id="S1oTDN0qhSM_"
# create training and testing splits
total_sentences = 62010
test_size = round(total_sentences / BATCH_SIZE * 0.2)
X_train = x_pad[BATCH_SIZE*test_size:]
Y_train = Y[BATCH_SIZE*test_size:]

X_test = x_pad[0:BATCH_SIZE*test_size]
Y_test = Y[0:BATCH_SIZE*test_size]
Y_train_int = tf.cast(Y_train, dtype=tf.int32)

train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train_int))
train_dataset = train_dataset.batch(BATCH_SIZE, drop_remainder=True)
```

```python id="zE_-LmOIhSM_" outputId="73abcb75-c390-4867-f37a-3d0bf8c75242"
loss_metric = tf.keras.metrics.Mean()

epochs = 5

# Iterate over epochs.
for epoch in range(epochs):
    print('Start of epoch %d' % (epoch,))

    # Iterate over the batches of the dataset.
    for step, (text_batch, labels_batch) in enumerate(train_dataset):
        labels_max = tf.argmax(labels_batch, -1, output_type=tf.int32)
        with tf.GradientTape() as tape:
            logits = blc_model(text_batch, training=True)
            loss = blc_model.crf.loss(labels_max, logits)

            grads = tape.gradient(loss, blc_model.trainable_weights)
            optimizer.apply_gradients(zip(grads, blc_model.trainable_weights))

            loss_metric(loss)
        if step % 50 == 0:
          print('step %s: mean loss = %s' % (step, loss_metric.result()))
```

```python id="CabFpZ1qhSNA"
Y_test_int = tf.cast(Y_test, dtype=tf.int32)

test_dataset = tf.data.Dataset.from_tensor_slices((X_test, Y_test_int))
test_dataset = test_dataset.batch(BATCH_SIZE, drop_remainder=True)
```

```python id="9yRGicv6hSNA"
out = blc_model.predict(test_dataset.take(1))
```

```python id="X37LSWohhSNA" outputId="16243d8c-a97b-4383-d83c-319499516fe4"
# check the outputs
print(out[1], tf.argmax(Y_test[1], -1))
print(out[2], tf.argmax(Y_test[2], -1))
```

```python id="hEaho_XVhSNA" outputId="1de9fc04-4629-4561-be1e-bbee6bfc8322"
text_tok.sequences_to_texts([X_test[2]])
```

```python id="XNztAYokhSNB" outputId="d83cf766-8a0d-4cca-ae2c-a5fae9140e92"
print("Ground Truth: ", ner_tok.sequences_to_texts([tf.argmax(Y_test[2], -1).numpy()]))
print("Prediction: ", ner_tok.sequences_to_texts([out[2]]))
```

```python id="pyXGejYshSNB" outputId="c59fab38-8f73-4366-87e3-e87911f24bf2"
print(ner_tok.sequences_to_texts([tf.argmax(Y_test[1], -1).numpy()]))
print(ner_tok.sequences_to_texts([out[1]]))
```

```python id="AeiEEXPdhSNB" outputId="65d56e10-53f7-4b04-d7f3-3c9ebba0bd4e"
blc_model.summary()
```

```python id="W9ZLhG7WhSNC"
def np_precision(pred, true):
    # expect numpy arrays
    assert pred.shape == true.shape
    assert len(pred.shape) == 2
    mask_pred = np.ma.masked_equal(pred, 0)
    mask_true = np.ma.masked_equal(true, 0)
    acc = np.equal(mask_pred, mask_true)
    return np.mean(acc.compressed().astype(int))
```

```python id="_WigWhk9hSNC" outputId="246632a6-2125-4fc9-a160-1d74409564ed"
np_precision(out, tf.argmax(Y_test[:BATCH_SIZE], -1).numpy())
```

```python id="Fpz5VOtrhSNC"

```
