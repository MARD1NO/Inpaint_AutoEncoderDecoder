import mxnet
from mxnet import gluon, init, nd, autograd, image
from mxnet.gluon import nn, data as gdata, loss as gloss, utils as gutils
import matplotlib.pyplot as plt
import time
import os

class AutoEncoder(nn.Block):
    def __init__(self, **kwargs):
        super(AutoEncoder, self).__init__(**kwargs)

        self.encoder_net = nn.Sequential()
        self.encoder_net.add(
            nn.Conv2D(16, kernel_size=3, strides=1, padding=1),
            nn.PReLU(),
            nn.Conv2D(16, kernel_size=4, strides=2, padding=1),
            nn.PReLU(),
            nn.Conv2D(32, kernel_size=3, strides=1, padding=1),
            nn.PReLU(),
            nn.Conv2D(32, kernel_size=4, strides=2, padding=1),
            nn.PReLU(),
            nn.Conv2D(64, kernel_size=3, strides=1, padding=1),
            nn.PReLU(),
            nn.Conv2D(64, kernel_size=4, strides=2, padding=1),
            nn.PReLU(),
            nn.Conv2D(128, kernel_size=4, strides=2, padding=1),
            nn.PReLU(),
            nn.Conv2D(128, kernel_size=4, strides=2, padding=1),
            nn.PReLU(),
            nn.Conv2D(256, kernel_size=3, strides=1, padding=1),
            nn.PReLU(),
            nn.Conv2D(256, kernel_size=4, strides=2, padding=1),
        ) # return <NDArray 1x256x4x4 @cpu(0)>,
        self.decoder_net = nn.Sequential()
        self.decoder_net.add(
            nn.Conv2DTranspose(256, kernel_size=3, strides=3, padding=2),
            nn.PReLU(),
            nn.Conv2DTranspose(128, kernel_size=3, strides=1, padding=1),
            nn.PReLU(),
            nn.Conv2DTranspose(128, kernel_size=2, strides=2),
            nn.PReLU(),
            nn.Conv2DTranspose(64, kernel_size=2, strides=2),
            nn.PReLU(),
            nn.Conv2DTranspose(64, kernel_size=2, strides=2),
            nn.PReLU(),
            nn.Conv2DTranspose(32, kernel_size=1, strides=1),
            nn.PReLU(),

            nn.Conv2DTranspose(32, kernel_size=2, strides=2),
            nn.PReLU(),
            nn.Conv2DTranspose(16, kernel_size=1, strides=1),
            nn.PReLU(),
            nn.Conv2DTranspose(16, kernel_size=2, strides=2),
            nn.PReLU(),
            nn.Conv2DTranspose(3, kernel_size=1, strides=1),
            nn.Activation('tanh')
        )   # return <NDArray 1x3x256x256 @cpu(0)>)

    def forward(self, x):
        encoded = self.encoder_net(x)
        decoded = self.decoder_net(encoded)
        return decoded
rgb_std = [0.485, 0.456, 0.406]
rgb_mean = [0.229, 0.224, 0.225]

def postpreprocess(img):
    return (img.transpose((1, 2, 0))*rgb_std + rgb_mean).clip(0, 1)


net = AutoEncoder()
net.initialize()

train_data = gdata.vision.ImageFolderDataset(r'./train')
normalize = gdata.vision.transforms.Normalize(
    [0.485, 0.456, 0.406],
    [0.229, 0.224, 0.225]
)

train_augs = gdata.vision.transforms.Compose([
    gdata.vision.transforms.Resize(256),
    gdata.vision.transforms.RandomFlipLeftRight(),
    gdata.vision.transforms.ToTensor(),
    normalize
])
batch_size = 20
max_epochs = 100


trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate':0.001})
ctx = mxnet.cpu()

def square_loss(Y_hat, Y):
    return (Y_hat[:, :, :, :] - Y[:, :, :, :]).square().mean()


class GetData(gdata.Dataset):
    def __init__(self, path1):
        super(GetData, self).__init__()
        self.path1 = path1
        # self.path2 = path2
        self.dataset1 = []
        # self.dataset2 = []
        self.dataset1.extend(os.listdir(path1))
        # self.dataset2.extend(os.listdir(path2))

    def __getitem__(self, idx):
        img_path1 = self.dataset1[idx]
        # img_path2 = self.dataset2[idx]

        img_1 = image.imread(os.path.join(self.path1, img_path1))
        # img_2 = image.imread(os.path.join(self.path2, img_path2))

        return img_1
            # , img_2

    def __len__(self):
        return len(self.dataset1)

dataset = GetData('./train/true')
dataset_2 = GetData('./train/masked')

normalize = gdata.vision.transforms.Normalize(
    [0.485, 0.456, 0.406],
    [0.229, 0.224, 0.225]
)

train_augs = gdata.vision.transforms.Compose([
    gdata.vision.transforms.Resize(256),
    gdata.vision.transforms.ToTensor(),
    normalize
])
batch_size = 20
max_epochs = 4000
train_iter = gdata.DataLoader(
    dataset.transform_first(train_augs), batch_size
)
masked_iter = gdata.DataLoader(
    dataset_2.transform_first(train_augs), batch_size
)

for epoch in range(max_epochs):
    train_l_sum, train_acc_sum, n, m, start = 0.0, 0.0, 0, 0, time.time()
    for true, masked in zip(train_iter, masked_iter):
        m += 1
        Xs, ys = masked, true
        with autograd.record():
            decoded = net(Xs)
            print("decoded!!!", decoded.shape)
            print(ys.shape)
            ls = square_loss(decoded, ys)
            print(ls.shape)
        ls.backward()
        trainer.step(batch_size)
        nd.waitall()
        train_l_sum += ls.sum().asscalar()
        n += len(ls)

        print('epoch %d, loss %.4f, time %.1f sec'
              % (epoch + 1, train_l_sum / n, time.time() - start))


        if m == 2:
            decode_img = decoded[0]
            plt.imsave('./decoded/decode_img_%d'%epoch, postpreprocess(decode_img.asnumpy()))
            true_img = ys[0]
            plt.imsave('./decoded/true_img_%d'%epoch, postpreprocess(true_img.asnumpy()))

    if epoch % 500 == 0:
        net.save_parameters('./ae_params/autoencoder_net_%d'%epoch)
