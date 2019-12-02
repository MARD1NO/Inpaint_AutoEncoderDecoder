import keras
from keras import layers
from keras import models
from keras.layers import Lambda
from keras import optimizers
from keras.utils import plot_model
import keras.backend as K
import keras_preprocessing.image as image
import numpy as np
import os
# 假设输入图像是64x64

def Generator_net(input):
    """
    生成器网络结构
    :param input: 输入张量，默认为64 x 64 x 3
    :return: 返回一个大小为 64x64x3的张量
    """

    x = layers.SeparableConv2D(filters=32, strides=(2, 2), kernel_size=(3, 3), padding='SAME' )(input)
    x = layers.LeakyReLU()(x)
    x = layers.BatchNormalization()(x)

    x = layers.SeparableConv2D(filters=64, strides=(2, 2), kernel_size=(3, 3), padding='SAME' )(x)
    x = layers.LeakyReLU()(x)
    x = layers.BatchNormalization()(x)

    x = layers.SeparableConv2D(filters=128, strides=(2, 2), kernel_size=(3, 3), padding='SAME' )(x)
    x = layers.LeakyReLU()(x)
    x = layers.BatchNormalization()(x)

    x = layers.SeparableConv2D(filters=256, strides=(2, 2), kernel_size=(3, 3), padding='SAME')(x)
    x = layers.LeakyReLU()(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2DTranspose(filters=256 , kernel_size=3, strides=(2, 2), padding='SAME')(x)
    x = layers.LeakyReLU()(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2DTranspose(filters=128, kernel_size=3, strides=(2, 2), padding='SAME')(x)
    x = layers.LeakyReLU()(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2DTranspose(filters=64, kernel_size=3, strides=(2, 2), padding='SAME')(x)
    x = layers.LeakyReLU()(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2DTranspose(filters=32, kernel_size=3, strides=(2, 2), padding='SAME')(x)
    x = layers.LeakyReLU()(x)
    x = layers.BatchNormalization()(x)

    x = layers.SeparableConv2D(filters=3, kernel_size=1, padding='SAME', activation='tanh')(x)

    generator = models.Model(input, x)
    # 绘制模型流程图
    # plot_model(generator, './generator.png', show_shapes=True)
    generator.summary()


    return generator


# outputs = Generator_net(inputs)
def resblock(input_tensor, filters):
    """
    定义一个简单的resnetBlock
    :param input_tensor: 输入张量
    :param filters: 滤波器个数
    :return: 经过一个resnetBlock的输出
    """
    conv_1 = layers.SeparableConv2D(int(filters//2), kernel_size=(3, 3), padding='SAME')(input_tensor)
    active_1 = Mish()(conv_1)
    batch_1 = layers.BatchNormalization()(active_1)

    conv_1_2 = layers.SeparableConv2D(filters, kernel_size=(3, 3), padding='SAME')(batch_1)
    active_1_2 = Mish()(conv_1_2)
    batch_1_2 = layers.BatchNormalization()(active_1_2)

    conv_2 = layers.SeparableConv2D(filters, kernel_size=(1, 1), padding='SAME')(input_tensor)
    active_2 = Mish()(conv_2)
    batch_2 = layers.BatchNormalization()(active_2)

    return layers.Add()([batch_1_2, batch_2])

def Mish():
    """
    定义mish激活函数
    :return: 返回一个Lambda层
    """
    return Lambda(lambda x:x*K.tanh(K.log(1 + K.exp(x))))

def Discriminator_net(input):
    """
    判别器网络
    :param input: 输入张量
    :return: 一个分类层输出的张量，用于分别是真图片还是假图片
    """
    x = resblock(input, filters=64)
    x = layers.AveragePooling2D()(x)
    x = resblock(x, filters=128)
    x = layers.AveragePooling2D()(x)
    x = resblock(x, filters=256)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(1, activation='sigmoid')(x)

    discriminator = models.Model(input, x)
    # 绘制模型流程图
    # plot_model(discriminator, './discriminator.png', show_shapes=True)
    discriminator.summary()
    discriminator_optimizer = optimizers.Adam(lr=0.0008,
                                              decay=1e-8)
    discriminator.trainable = False
    discriminator.compile(optimizer=discriminator_optimizer,
                          loss='mean_squared_error')

    return discriminator

# 定义GAN网络
discriminator_input = layers.Input(shape=(64, 64, 3)) # 定义一个输入占位符
generator_input = layers.Input(shape=(64, 64, 3)) # 定义一个输入占位符

discriminator = Discriminator_net(discriminator_input)
generator = Generator_net(generator_input)

inputs = keras.Input(shape=(64, 64, 3)) # 定义一个实际数据占位符
gan_output = discriminator(generator(inputs))

gan = keras.Model(inputs, gan_output)
# plot_model(gan, './gan.png', show_shapes=True)
gan_optimizer = optimizers.Adam(lr=0.0004, decay=1e-8)
gan.compile(optimizer=gan_optimizer, loss='binary_crossentropy')


imagegen = image.ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True
)
image_true_iter = imagegen.flow_from_directory(r'./train', class_mode=None, classes=None, target_size=(64, 64), batch_size=20,
                                          )

image_masked_iter = imagegen.flow_from_directory(r'./train/masked/', class_mode=None, classes=None, target_size=(64, 64), batch_size=20)

iteration = 1000
batch_size = 20
save_dir = './saved'
start = 0
latent_dim = (64, 64, 3)

for step in range(iteration):
    generated_images = generator.predict_generator(image_masked_iter, steps=20)

    stop = start + batch_size

    real_images = image_true_iter[start:stop]
    combined_images = np.concatenate([generated_images, real_images])

    labels = np.concatenate([np.ones((batch_size, 1)),
                             np.zeros((batch_size, 1))])

    labels += 0.05 * np.random.random(labels.shape)

    d_loss = discriminator.train_on_batch(combined_images, labels)

    random_latent_vectors = np.random.normal(size=(batch_size,latent_dim))

    misleading_targets = np.zeros((batch_size, 1))

    a_loss = gan.train_on_batch(random_latent_vectors,
                                misleading_targets)

    start += batch_size
    if start > len(generated_images) - batch_size:
        start = 0
    if step % 100 == 0:
        gan.save_weights('gan.h5')
    print('discriminator loss:', d_loss)
    print('adversarial loss:', a_loss)
    img = image.array_to_img(generated_images[0] * 255., scale=False)
    img.save(os.path.join(save_dir,
                          'generated_frog' + str(step) + '.png'))
    img = image.array_to_img(real_images[0] * 255., scale=False)
    img.save(os.path.join(save_dir,
                          'real_frog' + str(step) + '.png'))