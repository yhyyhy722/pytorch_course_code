import torch
import cv2 as cv
import numpy as np
import time
from age_gender_cnn import MyMulitpleTaskNet
from torchvision import transforms
model_bin = "D:/projects/face_detector/opencv_face_detector_uint8.pb";
config_text = "D:/projects/face_detector/opencv_face_detector.pbtxt";
genders = ['male', 'female']
transform = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                          std=[0.5, 0.5, 0.5]),
                                     transforms.Resize((64, 64))
                                     ])

def video_age_gender_demo():
    cnn_model = torch.load("./age_gender_model.pt")
    cnn_model.eval()
    print(cnn_model)
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
        # 绘制检测矩形 1x1xNx7
        for detection in cvOut[0,0,:,:]:
            score = float(detection[2])
            if score > 0.5:
                left = detection[3]*w
                top = detection[4]*h
                right = detection[5]*w
                bottom = detection[6]*h

                # age/gender model inference
                roi = frame[np.int32(top):np.int32(bottom),np.int32(left):np.int32(right),:]
                img = transform(roi)
                x_input = img.view(1, 3, 64, 64)
                # img = cv.resize(roi, (64, 64))
                # img = (np.float32(img) / 255.0 - 0.5) / 0.5
                # img = img.transpose((2, 0, 1))
                # x_input = torch.from_numpy(img).view(1, 3, 64, 64)
                age_, gender_ = cnn_model(x_input.cuda())
                predict_gender = torch.max(gender_, 1)[1].cpu().detach().numpy()[0]
                gender = "Male"
                if predict_gender == 1:
                    gender = "Female"
                predict_age = age_.cpu().detach().numpy()*116.0
                # print(predict_gender, predict_age)

                # 绘制
                cv.putText(frame, ("gender: %s, age:%d"%(gender, int(predict_age[0][0]))), (int(left), int(top)-15), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 1)
                cv.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (255, 0, 0), thickness=2)
        c = cv.waitKey(1)
        if c == 27:
            break
        cv.imshow("age and gender demo", frame)

    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    video_age_gender_demo()