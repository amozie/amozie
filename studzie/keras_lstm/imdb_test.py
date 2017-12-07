from keras.datasets import imdb
from keras.preprocessing import sequence, text
from keras.layers import *
from keras.models import *
from keras.callbacks import TensorBoard

top_words = 5000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)

max_review_length = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)

embedding_vector_length = 32
model = Sequential()
model.add(Embedding(top_words, embedding_vector_length, input_length=max_review_length))
model.add(GRU(64))
model.add(Dense(1, activation='sigmoid'))
model.compile('adam', 'binary_crossentropy', ['accuracy'])
model.summary()

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=1, batch_size=64,
          callbacks=[TensorBoard('c:/users/dell/appdata/local/temp/tf.log')])

model.evaluate(X_test, y_test)
