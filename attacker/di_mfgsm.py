import tensorflow as tf
import numpy as np
from keras import backend as K


def create_one_attack_pic(model, img):
    # original_image = image.img_to_array(img)
    #print(img)
    hacked_image = np.copy(img)
    hacked_image = np.expand_dims(hacked_image, 0)

    actual_class = np.argmax(model.predict(hacked_image)[0])
    
    # fake_class = 0
    # if actual_class == 9:
    #     fake_class = 0
    # else:
    #     fake_class= actual_class + 1

    model_input_layer = model.layers[0].input
    model_output_layer = model.layers[-1].output

    cost_function = model_output_layer[0, actual_class]
    gradient_function = K.gradients(cost_function, model_input_layer)[0]
    grab_cost_and_gradients_from_model = K.function([model_input_layer, K.learning_phase()],
                                                    [cost_function, gradient_function])
    e = 0.015
    index = 1
    delta = 0.04
    max_change_above = hacked_image + delta
    max_change_below = hacked_image - delta
    old_gradients = 0
    sure_count = 0
    while True:
        cost, gradients = grab_cost_and_gradients_from_model([input_diversity(hacked_image), 0])
        if cost <0.01 and index >= 10 or cost == 0: 
            # sure_count += 1
            # if sure_count >= 2:
                break
        # else:
        #     sure_count = 0
        gradients = get_gradient(gradients, old_gradients)
        old_gradients = gradients
        if index % 3 == 0:
            max_change_above = max_change_above + delta
            max_change_below = max_change_below - delta
        n = np.sign(gradients)
        #n=gradients
        hacked_image -= n * e
        hacked_image = np.clip(hacked_image, max_change_below, max_change_above)
        hacked_image = np.clip(hacked_image, 0, 1.0)
        index += 1
    print("batch:{} Cost:{:.8}%".format(index, cost * 100))
    # print(hacked_image)
    return hacked_image[0]


def get_gradient(gradients, old_gradients):
    a = 0.3
    # print(gradients.shape)
    return old_gradients * a + gradients / np.linalg.norm(gradients[0, :, :, 0], ord=1)
    #return gradients


def input_diversity(input_tensor):
    return tf.cond(tf.random_uniform(shape=[1])[0] < tf.constant(0.5), lambda: change_shape(input_tensor), lambda: input_tensor)
    # return input_tensor

def change_shape(input_tensor):
    lowWidth = 22
    originWidth = 28
    rnd = tf.random_uniform((), lowWidth, originWidth, dtype=tf.int32)
    #rnd=26
    rescaled = tf.image.resize_images(input_tensor, [rnd, rnd], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    h_rem = originWidth - rnd
    w_rem = originWidth - rnd
    pad_top = tf.random_uniform((), 0, h_rem, dtype=tf.int32)
    #pad_top=1
    pad_bottom = h_rem - pad_top
    pad_left = tf.random_uniform((), 0, w_rem, dtype=tf.int32)
    #pad_left=1
    pad_right = w_rem - pad_left
    padded = tf.pad(rescaled, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], constant_values=0.)
    padded.set_shape((input_tensor.shape[0], originWidth, originWidth, 1))
    return padded