#1. 데이터 불러오기
import pandas as pd
import numpy as np

df_feature = pd.read_csv("onenavi_train_feature.csv", sep="|")
df_target = pd.read_csv("onenavi_train_target.csv", sep="|")

from sklearn.model_selection import train_test_split

train_x, test_x, train_y, test_y = train_test_split(df_feature, df_target, test_size= 0.20, random_state= 42)

#2. 회귀 문제에 딥러닝 모델 적용하기
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.callbacks import ModelCheckpoint, EarlyStopping

##2.1. 모델링
def build_model():
  model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=[len(train_x.keys())]),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
  ])

  model.compile(loss='mse',
                optimizer='adam',
                metrics=['mae', 'mse'])
  return model

model = build_model()
model.summary()

tf.keras.utils.plot_model(model, show_shapes=True)

early_stopping = EarlyStopping(monitor='val_loss', patience=10)

checkpoint_path = 'tmp_checkpoint.ckpt'
cb_checkpoint = ModelCheckpoint(checkpoint_path, save_weights_only=True, monitor='val_loss',
                               verbose=1, save_best_only=True)

history = model.fit(train_x, train_y, epochs=30,  
                   validation_data = (test_x,test_y),
                    callbacks=[cb_checkpoint, early_stopping]
                    )

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 30), history.history["loss"], label="train_loss")
plt.plot(np.arange(0, 30), history.history["val_loss"], label="val_loss")
plt.title("Training Loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend()
plt.show()

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 30), history.history["mae"], label="train_mae")
plt.plot(np.arange(0, 30), history.history["val_mae"], label="val_mae")
plt.title("Training mae")
plt.xlabel("Epoch #")
plt.ylabel("mae")
plt.legend()
plt.show()

##2.2 최적 모델 불러오기 및 저장
model.load_weights(checkpoint_path)
model.save("DeeplearningModel.h5")

##2.3 모델 이해하기
model.layers[0].get_weights()[0] # Input_layer 가중치
model.layers[0].get_weights()[1] # Input_layer 편향
model.layers[1].get_weights()[0] # Dense 가중치
model.layers[1].get_weights()[1] # Dense 편향
model.layers[2].get_weights()[0] # output 가중치
model.layers[2].get_weights()[1] # output 편향

