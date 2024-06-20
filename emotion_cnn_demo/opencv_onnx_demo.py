import cv2 as cv
import numpy as np
emotion_labels = ["neutral","anger","disdain","disgust","fear","happy","sadness","surprise"]
# import pyttsx3

def mnist_onnx_demo():
    mnist_net = cv.dnn.readNetFromONNX("cnn_mnist.onnx")
    image = cv.imread("D:/images/9_99.png")
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    cv.imshow("input", gray)
    # HW -> NCHW
    blob = cv.dnn.blobFromImage(gray, 0.00392, (28, 28), (127.0)) / 0.5
    print(blob.shape)
    mnist_net.setInput(blob)
    result = mnist_net.forward()
    pred_label = np.argmax(result, 1)
    print("predit label : %d"%pred_label)
    # engine = pyttsx3.init()
    # engine.say(str(pred_label))
    # engine.runAndWait()
    cv.waitKey(0)
    cv.destroyAllWindows()


def landmark_onnx_demo():
    landmark_net = cv.dnn.readNetFromONNX("landmarks_cnn.onnx")
    image = cv.imread("D:/bird_test/325.jpg")
    cv.imshow("input", image)
    h, w, c = image.shape
    blob = cv.dnn.blobFromImage(image, 0.00392, (64, 64), (127, 127, 127), False) / 0.5
    landmark_net.setInput(blob)
    lm_pts = landmark_net.forward()
    lm_pts = np.reshape(lm_pts, (5, 2))
    print(lm_pts)
    for x, y in lm_pts:
        print(x, y)
        x1 = x * w
        y1 = y * h
        cv.circle(image, (np.int32(x1), np.int32(y1)), 2, (0, 0, 255), 2, 8, 0)
    cv.imshow("face landmark detection", image)
    cv.imwrite("D:/landmark_det_result.png", image)
    cv.waitKey(0)
    cv.destroyAllWindows()


def emotions_onnx_demo():
    emotion_net = cv.dnn.readNetFromONNX("face_emotions_model.onnx")
    image = cv.imread("D:/facedb/test/367.jpg")
    blob = cv.dnn.blobFromImage(image, 0.00392, (64, 64), (127, 127, 127), False) / 0.5
    emotion_net.setInput(blob)
    res = emotion_net.forward("output")
    idx = np.argmax(np.reshape(res, (8)))
    emotion_txt = emotion_labels[idx]
    cv.putText(image, emotion_txt, (10, 25), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv.imshow("emotion detection", image)

    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    mnist_onnx_demo()