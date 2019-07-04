
import numpy as np
import tensorflow as tf
from skimage.measure import compare_ssim as ssim
import keras


# import matplotlib.pyplot as plt

def compare_images(imageA, imageB):
    s = ssim(imageA, imageB, multichannel=True)
    return s


class DataSet():
    def __init__(self):
        self.fashion_mnist = keras.datasets.fashion_mnist
        (self.train_images, self.train_labels), (self.test_images, self.test_labels) = self.fashion_mnist.load_data()
        self.class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        
        self.train_images = self.train_images / 255.0
        self.test_images = self.test_images / 255.0

        #self.train_images = np.array(self.train_images)
        #self.test_images = np.array(self.test_images)

    def reshape(self, shape):
        self.train_images = self.train_images.reshape(self.train_images.shape[0], *shape)
        self.test_images = self.test_images.reshape(self.test_images.shape[0], *shape)


    # def plotTestData(self, index):
    #     self.plotImg(self.test_images[index])
    #
    # def plotImg(self, img):
    #     plt.figure()
    #     plt.imshow(img)
    #     plt.colorbar()
    #     plt.grid(False)
    #     plt.show()


class Model():
    def __init__(self, inputShape=(28, 28, 1)):
        self.model = keras.Sequential([
            keras.layers.Flatten(input_shape=inputShape, name="flatten"),
            keras.layers.Dense(128, activation=tf.nn.relu, name="dense1"),
            keras.layers.Dense(10, activation=tf.nn.softmax, name="output")
        ])
        self.epochs = 5
        self.model_file_path = "models/model.json"
        self.model_weight_file_path = "models/model.h5"
        pass

    def save(self):
        model_json = self.model.to_json()
        with open(self.model_file_path, "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights(self.model_weight_file_path)
        print("Saved model to disk")

    def load(self):
        json_file = open(self.model_file_path, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = keras.models.model_from_json(loaded_model_json)
        # load weights into new model
        self.loadWeights()
        print("Loaded model from disk")

    def loadWeights(self):
        self.model.load_weights(self.model_weight_file_path)

    def train(self, dataset):
        self.model.compile(optimizer='adam', 
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
        self.model.fit(dataset.train_images, dataset.train_labels, epochs=self.epochs)

    def test(self, dataset):
        self.model.compile(optimizer='adam', 
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
        test_loss, test_acc = self.model.evaluate(dataset.test_images, dataset.test_labels)
        print('Test accuracy:', test_acc)

    def predict(self, img):
        # 返回预测的分类标签
        img = (np.expand_dims(img,0))        
        return np.argmax((self.model.predict(img))[0])

    def testPrediction(self, img, trueLabel):
        # 返回预测的对不对
        return self.predict(img) == trueLabel

    def testAttackResult(self, imgs, adversarialImgs, labels):
        success_count = 0
        ssim_list = []
        for i in range(len(imgs)):
            prediction = self.testPrediction(adversarialImgs[i], labels[i])
            ssim_list.append(compare_images(imgs[i], adversarialImgs[i]))
            if prediction == False:
                print("pic" + str(i) + "success!")
                success_count = success_count + 1
            else:
                print("pic" + str(i) + "fail!")
        print("success rate: ", str(success_count/len(imgs)))
        print("average ssim: ", str(sum(ssim_list)/len(ssim_list)))
#
# class Model_CNN(Model):
#     def __init__(self, inputShape=(28, 28, 1)):
#         self.model = keras.Sequential()
#         self.model.add(Conv2D(32, kernel_size=(3, 3),
#                  activation='relu',
#                  input_shape=inputShape))
#         self.model.add(Conv2D(64, (3, 3), activation='relu'))
#         self.model.add(MaxPooling2D(pool_size=(2, 2)))
#         self.model.add(Dropout(0.25))
#         self.model.add(Flatten())
#         self.model.add(Dense(128, activation='relu'))
#         self.model.add(Dropout(0.5))
#         self.model.add(Dense(10, activation='softmax'))
#
#         self.epochs = 5
#         self.model_file_path = "model_cnn.json"
#         self.model_weight_file_path = "model_cnn.h5"
#         pass


#
# class AttackModel(Model):
#     def __init__(self, targetModel, inputShape=(28, 28)):
#         self.model = keras.Sequential([
#             keras.layers.Flatten(input_shape=inputShape, name="flatten"),
#             Layer.DisturbanceLayer(name="disturbance"),
#             keras.layers.Dense(128, activation=tf.nn.relu, name="dense1"),
#             keras.layers.Dense(10, activation=tf.nn.softmax, name="output")
#         ])
#
#         self.model.get_layer("flatten").trainable = False
#         dense_1_layer = self.model.get_layer("dense1")
#         dense_1_layer.set_weights(targetModel.model.get_layer("dense1").get_weights())
#         dense_1_layer.trainable = False
#         output_layer = self.model.get_layer("output")
#         output_layer.set_weights(targetModel.model.get_layer("output").get_weights())
#         output_layer.trainable = False
#
#         self.adverseFunction = K.function([self.model.input],[self.model.get_layer("disturbance").output])
#         self.epochs = 50
#         self.model_file_path = "model_attack.json"
#         self.model_weight_file_path = "model_attack.h5"
#
#
#     def train(self, dataset):
#         # earlystopping = EarlyStopping(monitor='acc', baseline=1.0)
#         self.model.compile(optimizer='adam',
#                 loss='sparse_categorical_crossentropy',
#                 #loss=custom_loss(),
#                 metrics=['accuracy']
#                )
#         # self.model.fit(dataset['train_images'], dataset['train_labels'], epochs=self.epochs, callbacks=[earlystopping])
#         self.model.fit(dataset['train_images'], dataset['train_labels'], epochs=self.epochs, verbose=0)
#
#     def getAdversarialImg(self, img):
#         img = (np.expand_dims(img,0))
#         # adversarialModel = keras.models.Model(inputs=self.model.input, outputs=self.model.get_layer("disturbance").output)
#         # adversarialImg = adversarialModel.predict(img)[0]
#
#         adversarialImg = self.adverseFunction([img])[0]
#         adversarialImg = tf.reshape(adversarialImg, [28, 28])
#
#         return K.eval(adversarialImg)


'''
# Define custom loss
def custom_loss():

    # Create a loss function that adds the MSE loss to the mean of all squared activations of a specific layer
    def loss(y_true,y_pred):
        print("yyyyTRUE____")

        if y_true[0] == np.argmax(y_pred[0]):
            return 0
        cross = K.sparse_categorical_crossentropy(y_true, y_pred)
        return cross
   
    # Return a function
    return loss
'''
