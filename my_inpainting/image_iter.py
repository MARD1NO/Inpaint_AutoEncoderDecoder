from keras_preprocessing import image
import os
def train_iter(dir):
    train = []
    for i in os.listdir(dir):
        train.append(os.path.join(dir, i))
        yield train


# a = train_iter('./train/true')
# print(list(a))