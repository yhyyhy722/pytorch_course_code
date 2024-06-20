import torch
import torch.nn.functional as F
from emotions_dataset import EmotionDataset
from torch.utils.data import DataLoader

# 检查是否可以利用GPU
train_on_gpu = torch.cuda.is_available()
if not train_on_gpu:
    print('CUDA is not available.')
else:
    print('CUDA is available!')


class ResidualBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        """
        Args:
          in_channels (int):  Number of input channels.
          out_channels (int): Number of output channels.
          stride (int):       Controls the stride.
        """
        super(ResidualBlock, self).__init__()

        self.skip = torch.nn.Sequential()

        if stride != 1 or in_channels != out_channels:
            self.skip = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, bias=False),
                torch.nn.BatchNorm2d(out_channels))

        self.block = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=stride, bias=False),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=1, bias=False),
            torch.nn.BatchNorm2d(out_channels))

    def forward(self, x):
        out = self.block(x)
        identity = self.skip(x)
        out += identity
        out = F.relu(out)
        return out


class EmotionsResNet(torch.nn.Module):
    def __init__(self):
        super(EmotionsResNet, self).__init__()
        self.cnn_layers = torch.nn.Sequential(
            # 卷积层 (64x64x3)
            ResidualBlock(3, 32, 1),
            ResidualBlock(32, 64, 2),
            ResidualBlock(64, 64, 2),
            ResidualBlock(64, 128, 2),
            ResidualBlock(128, 128, 2),
            ResidualBlock(128, 256, 2),
            ResidualBlock(256, 256, 2)
        )
        self.cb1x1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=256, out_channels=8, kernel_size=1, padding=0, stride=1,
                            bias=False),
            torch.nn.BatchNorm2d(8),
            torch.nn.ReLU()
            )

    def forward(self, x):
        # stack convolution layers
        x = self.cnn_layers(x)
        x = self.cb1x1(x)
        # Nx8x1x1
        B, C, H, W = x.size()
        out = x.view(B, -1)
        return out


if __name__ == "__main__":
    # create a complete CNN
    model = EmotionsResNet()
    print(model)

    # 使用GPU
    if train_on_gpu:
        model.cuda()

    ds = EmotionDataset("D:/facedb/emotion_dataset")
    num_train_samples = ds.num_of_samples()
    bs = 16
    dataloader = DataLoader(ds, batch_size=bs, shuffle=True)

    # 训练模型的次数
    num_epochs = 15
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    model.train()

    # 损失函数
    cross_loss = torch.nn.CrossEntropyLoss()
    index = 0
    for epoch in  range(num_epochs):
        train_loss = 0.0
        for i_batch, sample_batched in enumerate(dataloader):
            images_batch, emotion_batch = \
                sample_batched['image'], sample_batched['emotion']
            if train_on_gpu:
                images_batch, emotion_batch= images_batch.cuda(), emotion_batch.cuda()
            optimizer.zero_grad()

            # forward pass: compute predicted outputs by passing inputs to the model
            m_emotion_out_ = model(images_batch)
            emotion_batch = emotion_batch.long()

            # calculate the batch loss
            loss = cross_loss(m_emotion_out_, emotion_batch)

            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()

            # perform a single optimization step (parameter update)
            optimizer.step()

            # update training loss
            train_loss += loss.item()
            if index % 100 == 0:
                print('step: {} \tTraining Loss: {:.6f} '.format(index, loss.item()))
            index += 1

            # 计算平均损失
        train_loss = train_loss / num_train_samples

        # 显示训练集与验证集的损失函数
        print('Epoch: {} \tTraining Loss: {:.6f} '.format(epoch, train_loss))

    # save model
    model.eval()
    torch.save(model, 'face_emotions_model.pt')
