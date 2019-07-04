import keras
import numpy as np
import tensorflow as tf
from keras import backend as K
from main import aiTest
from models import load_model
from skimage.measure import compare_ssim as ssim
from models.basicModel import Model

img_rows, img_cols = 28, 28
num_classes = 10

fashion_mnist = keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# TODO here to change test models
#loaded_model = load_model()
# model = Model()
# model.load()
# loaded_model = model.model
loaded_model=tf.keras.models.load_model("models/mnist_model.h5")

test_pic_num = 100
x_test = x_test[:test_pic_num]
y_test = y_test[:test_pic_num]
attack_sample = aiTest(x_test, (28, 28, 1))
count = 0.0
s = 0.0
for i in range(0, test_pic_num):
    origin_img = np.expand_dims(x_test[i], 0)[:,:,:,0]
    attack_img = np.expand_dims(attack_sample[i], 0)[:,:,:,0]
    # origin_img = np.expand_dims(x_test[i], 0)
    # attack_img = np.expand_dims(attack_sample[i], 0)
    actual_class = np.argmax(loaded_model.predict(origin_img)[0])
    # print("actual: ", actual_class)
    wrong_class = np.argmax(loaded_model.predict(attack_img)[0])
    # wrong_class=np.argmax(l_model.predict(origin_img)[0])
    # print("wrong: ", wrong_class)
    #s += ssim(origin_img[0,:,:,0], attack_img[0,:,:,0], multichannel=True) / test_pic_num
    s += ssim(origin_img[0], attack_img[0], multichannel=True) / test_pic_num
    if actual_class != wrong_class:
        count += 1
print(count / test_pic_num)
print("ssim:", s)
