"""
Cryptocurrencies future price movement predictor recurrent neural network:
based on Crypto currencies closing price and volume.

Tensorboard is used to visulize the epoch data, accuracy, loss etc.
You can run tensorboard by running below command at the project root.
`tensorboard --logdir=logs/
"""
import Get_data
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import tensorflow as tf
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization
from tensorflow.keras.callbacks import History, LearningRateScheduler, ModelCheckpoint

# EPOCHS = 100  # how many passes through our data
# BATCH_SIZE = 16  # how many batches? Try smaller batch if you're getting OOM (out of memory) errors.
# NAME = f"{ticker}-SEQ-{FUTURE_PERIOD_PREDICT}-PRED-{int(time.time())}"
# NAME = f"{SEQ_LEN}-SEQ-{FUTURE_PERIOD_PREDICT}-PRED-{int(time.time())}"
# print(NAME)

learning_rate = .1e-05
decay_rate = .008



def plot_loss(loss_history):
	fig, ax = plt.subplots(1, 1, figsize=(10,6))
	ax.plot(np.sqrt(loss_history.history['loss']), 'r', label='train')
	ax.plot(np.sqrt(loss_history.history['val_loss']), 'b' ,label='val')
	ax.set_xlabel(r'Epoch', fontsize=20)
	ax.set_ylabel(r'Loss', fontsize=20)
	ax.legend()
	ax.tick_params(labelsize=20)
	plt.show()
	# Plot the accuracy
	fig, ax = plt.subplots(1, 1, figsize=(10,6))
	ax.plot(np.sqrt(loss_history.history['accuracy']), 'r', label='train')
	ax.plot(np.sqrt(loss_history.history['val_accuracy']), 'b' ,label='val')
	ax.set_xlabel(r'Epoch', fontsize=20)
	ax.set_ylabel(r'Accuracy', fontsize=20)
	ax.legend()
	ax.tick_params(labelsize=20)
	plt.show()

def train_model(lookback, epochs=100, batch_size=16, init_mode='uniform'):
	# tensorboard = TensorBoard(log_dir=f"{os.getcwd()}/logs/{ticker}")
	# checkpoint = ModelCheckpoint("{}/Data/{}.model_check".format(os.getcwd(), ticker, monitor='val_accuracy',
	# verbose=1, save_best_only=True, mode='max')) # saves only the best ones
	model = Sequential()
	model.add(LSTM(512, input_shape=(lookback, 1), return_sequences=True)) # previous value was 128
	model.add(Dropout(0.2))
	model.add(BatchNormalization())

	model.add(LSTM(256, return_sequences=True)) # previous value was 128
	model.add(Dropout(0.1))
	model.add(BatchNormalization())

	model.add(LSTM(128))
	model.add(Dropout(0.2))
	model.add(BatchNormalization())

	model.add(Dense(64, kernel_initializer=init_mode, activation='relu')) # default tanh for LSTM but using relu, previous value was 32, dropout .2
	model.add(Dropout(0.1))

	model.add(Dense(2, kernel_initializer=init_mode, activation='softmax'))
	opt = tf.keras.optimizers.Adam(lr=learning_rate, decay=decay_rate)

	model.compile(
		loss='sparse_categorical_crossentropy',
		optimizer=opt,
		metrics=['accuracy'])

	# loss_history = History()
	# lr_rate = LearningRateScheduler(exp_decay)
	# callbacks_list = [loss_history, lr_rate]

	# filepath = "RNN_Final-{epoch:02d}-{val_accuracy:.3f}"  # unique file name that will include the epoch and the validation acc for that epoch
	# checkpoint = ModelCheckpoint("{}/Data/{}.model".format(os.getcwd(), filepath, monitor='val_accuracy', verbose=1,
	# save_best_only=True, mode='max')) # saves only the best ones
	
	# model.fit(
	# 	train_x, train_y,
	# 	batch_size=batch_size,
	# 	epochs=epochs,
	# 	validation_data=(validation_x, validation_y),
	# 	callbacks=callbacks_list # [tensorboard, checkpoint]
	# )
	# plot_loss(loss_history)
	return model

def exp_decay(epoch):
    lrate = learning_rate * np.exp(-decay_rate*epoch)
    return lrate

def predict_lstm(model, train_x, train_y, validation_x, validation_y, live):
	train_x = np.array(train_x)
	train_y = np.array(train_y)
	validation_x = np.array(validation_x)
	validation_y = np.array(validation_y)

	if not live:
		score = model.evaluate(validation_x, validation_y, verbose=0)
		print('Test loss:', score[0])
		print('Test accuracy:', score[1])
	y_pred = model.predict(validation_x)	
	return y_pred