import torch
import torch.nn.functional as F
from capcha.capcha_dataset import output_nums, CapchaDataset
from torch.utils.data import DataLoader

# 检查是否可以利用GPU
train_on_gpu = torch.cuda.is_available()
if not train_on_gpu:
    print('CUDA is not available.')
else:
    print('CUDA is available!')


class ResidualBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
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


class CapchaResNet(torch.nn.Module):
    def __init__(self):
        super(CapchaResNet, self).__init__()
        self.cnn_layers = torch.nn.Sequential(
            # 卷积层 (128x32x3)
            ResidualBlock(3, 32, 1),
            ResidualBlock(32, 64, 2),
            ResidualBlock(64, 64, 2),
            ResidualBlock(64, 128, 2),
            ResidualBlock(128, 256, 2),
            ResidualBlock(256, 256, 2),
        )

        self.fc_layers = torch.nn.Sequential(
            torch.nn.Linear(256 * 4, output_nums()),
        )

    def forward(self, x):
        # resnet convolution layers
        x = self.cnn_layers(x)
        out = x.view(-1, 4 * 256)
        out = self.fc_layers(out)
        return out


if __name__ == "__main__":
    # create a complete CNN
    model = CapchaResNet()
    print(model)

    # 使用GPU
    if train_on_gpu:
        model.cuda()

    ds = CapchaDataset("D:/python/pytorch_tutorial/capcha/samples")
    num_train_samples = ds.num_of_samples()
    bs = 32
    dataloader = DataLoader(ds, batch_size=bs, shuffle=True)

    # 训练模型的次数
    num_epochs = 42
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    model.train()

    # 损失函数
    mul_loss = torch.nn.MultiLabelSoftMarginLoss()
    index = 0
    for epoch in range(num_epochs):
        train_loss = 0.0
        for i_batch, sample_batched in enumerate(dataloader):
            images_batch, oh_labels = \
                sample_batched['image'], sample_batched['encode']
            if train_on_gpu:
                images_batch, oh_labels= images_batch.cuda(), oh_labels.cuda()
            optimizer.zero_grad()

            # forward pass: compute predicted outputs by passing inputs to the model
            m_label_out_ = model(images_batch)
            oh_labels = torch.autograd.Variable(oh_labels.float())

            # calculate the batch loss
            loss = mul_loss(m_label_out_, oh_labels)

            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()

            # perform a single optimization step (parameter update)
            optimizer.step()

            # update training loss
            train_loss += loss.item()

            # 计算平均损失
        train_loss = train_loss / num_train_samples

        # 显示训练集与验证集的损失函数
        print('Epoch: {} \tTraining Loss: {:.6f} '.format(epoch, train_loss))

    # save model
    model.eval()
    torch.save(model, 'capcha_recog_model.pt')
