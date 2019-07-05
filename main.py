import numpy as np
from models import load_model
from attacker import create_one_attack_pic as create_sample
from models import basicModel


def aiTest(images, shape):
    result = []
    model = get_basic_model()
    model2=get_basic_model2()
    images=images/255
    for i in images:
        img1 = create_sample(model, i)
        #img2 = create_sample(model2, i)
        #img = (img1 + img2) / 2 
        result.append(img1)
    result = np.asarray(result)*255
    return result


def get_basic_model():
    # TODO change model here
    # model = load_model()
    # return model
    model = basicModel.Model()
    model.load()
    return model.model

def get_basic_model2():
    model = load_model()
    return model