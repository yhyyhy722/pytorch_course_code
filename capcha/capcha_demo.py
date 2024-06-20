import cv2 as cv
import numpy as np
import torch
import os
from capcha.capcha_model import ResidualBlock, CapchaResNet
NUMBER = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
ALPHABET = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
ALL_CHAR_SET = NUMBER + ALPHABET
ALL_CHAR_SET_LEN = len(ALL_CHAR_SET)
MAX_CAPTCHA = 5


def test_capcha_codes():
    cnn_model = torch.load("./capcha_recog_model.pt")
    cnn_model.eval()
    root_dir = "D:/python/pytorch_tutorial/capcha/testdata"
    files = os.listdir(root_dir)
    one_hot_len = ALL_CHAR_SET_LEN
    for file in files:
        if os.path.isfile(os.path.join(root_dir, file)):
            image = cv.imread(os.path.join(root_dir, file))
            h, w, c = image.shape
            img = cv.resize(image, (128, 32))
            img = (np.float32(img) /255.0 - 0.5) / 0.5
            img = img.transpose((2, 0, 1))
            x_input = torch.from_numpy(img).view(1, 3, 32, 128)
            probs = cnn_model(x_input.cuda())
            mul_pred_labels = probs.squeeze().cpu().tolist()
            c0 = ALL_CHAR_SET[np.argmax(mul_pred_labels[0:one_hot_len])]
            c1 = ALL_CHAR_SET[np.argmax(mul_pred_labels[one_hot_len:one_hot_len*2])]
            c2 = ALL_CHAR_SET[np.argmax(mul_pred_labels[one_hot_len*2:one_hot_len*3])]
            c3 = ALL_CHAR_SET[np.argmax(mul_pred_labels[one_hot_len*3:one_hot_len*4])]
            c4 = ALL_CHAR_SET[np.argmax(mul_pred_labels[one_hot_len*4:one_hot_len*5])]
            pred_txt = '%s%s%s%s%s' % (c0, c1, c2, c3, c4)
            cv.putText(image, pred_txt, (10, 20), cv.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 2)
            print("current code : %s, predict code : %s "%(file[:-4], pred_txt))
            cv.imshow("capcha predict", image)
            cv.waitKey(0)


if __name__ == "__main__":
    test_capcha_codes()
