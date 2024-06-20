import cv2 as cv
import numpy as np
import torch
from emotions_cnn import ResidualBlock, EmotionsResNet
emotion_labels = ["neutral","anger","disdain","disgust","fear","happy","sadness","surprise"]
model_bin = "D:/projects/face_detector/opencv_face_detector_uint8.pb";
config_text = "D:/projects/face_detector/opencv_face_detector.pbtxt";


def video_emotion_demo():
    cnn_model = torch.load("./face_emotions_model.pt")
    # capture = cv.VideoCapture(0)
    capture = cv.VideoCapture("D:/images/video/example_dsh.mp4")

    # load tensorflow model
    net = cv.dnn.readNetFromTensorflow(model_bin, config=config_text)
    while True:
        ret, frame = capture.read()
        if ret is not True:
            break
        h, w, c = frame.shape
        blobImage = cv.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0), False, False);
        net.setInput(blobImage)
        cvOut = net.forward()
        # 绘制检测矩形
        for detection in cvOut[0,0,:,:]:
            score = float(detection[2])
            if score > 0.5:
                left = detection[3]*w
                top = detection[4]*h
                right = detection[5]*w
                bottom = detection[6]*h

                # roi and detect emotion
                roi = frame[np.int32(top):np.int32(bottom),np.int32(left):np.int32(right),:]
                img = cv.resize(roi, (64, 64))
                img = (np.float32(img) / 255.0 - 0.5) / 0.5
                img = img.transpose((2, 0, 1))
                x_input = torch.from_numpy(img).view(1, 3, 64, 64)
                probs = cnn_model(x_input.cuda())
                em_idx = torch.max(probs, 1)[1].cpu().detach().numpy()[0]

                # 绘制
                cv.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (255, 0, 0), thickness=2)
                cv.putText(frame, emotion_labels[em_idx], (int(left), int(top)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        c = cv.waitKey(1)
        if c == 27:
            break
        cv.imshow("face detection + emotion", frame)

    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    video_emotion_demo()