import tensorflow
from tensorflow import keras
import numpy
data = keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels) = data.load_data(num_words=88000)

word_index = data.get_word_index()

word_index = {k: (v+3) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

train_data  = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index["<PAD>"], padding="post", maxlen=250 )
test_data  = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index["<PAD>"], padding="post", maxlen=250 )


def review_encode(s):
    encoded = [1]
    for word in s:
        if word.lower() in word_index:
            encoded.append(word_index[word.lower()])
        else:
            encoded.append(2)
    return encoded

# Loading model here
model = keras.models.load_model("Text_classification.h5")
with open("review_test.txt", encoding="utf-8") as f:
    for line in f.readlines():
        nline = line.replace(",", "").replace(".", "").replace(")", "").replace(":", "").replace("\"","").strip().split(" ")
        encode = review_encode(nline)
        encode = keras.preprocessing.sequence.pad_sequences([encode], value=word_index["<PAD>"], padding="post", maxlen=250)
        predict = model.predict(encode)
        #print(line)
        #print(encode)
        print(predict[0])


if predict[0] > 0.7:
    print("The review was positive")

elif predict[0]<0.7 and predict[0]>0.4:
    print("That was a fair review")

else:
    print("That was a bad review")

