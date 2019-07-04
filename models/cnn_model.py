from keras.models import model_from_json


def load_model():
    json_file = open('models/model_cnn.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("models/model_cnn.h5")
    return loaded_model
