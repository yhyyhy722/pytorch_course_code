from __future__ import print_function
import cv2 as cv
import numpy as np
import torch
from openvino.runtime import Core
from vehicle_attributes_cnn import ResidualBlock, VehicleAttributesResNet

color_labels = ["white", "gray", "yellow", "red", "green", "blue", "black"]
type_labels = ["car", "bus", "truck", "van"]

model_xml = "D:/projects/vehicle-detection-0202.xml"
model_bin = "D:/projects/vehicle-detection-0202.bin"

ie = Core()
model = ie.read_model(model=model_xml)
detect_model = ie.compile_model(model=model, device_name="CPU")
output_layer = detect_model.output(0)

cnn_model = torch.load("./vehicle_attributes_resnet.pt")
print(cnn_model)

capture = cv.VideoCapture("D:/images/video/cars-1900.mp4")
ih = capture.get(cv.CAP_PROP_FRAME_HEIGHT)
iw = capture.get(cv.CAP_PROP_FRAME_WIDTH)
cv.namedWindow("Vehicle Attributes Recognition Demo", cv.WINDOW_NORMAL)

while True:
    ret, frame = capture.read()
    if ret is not True:
        break
    h, w, c = frame.shape
    blob = cv.dnn.blobFromImage(frame, 1.0, (512, 512), (0, 0, 0), False, False)
    res = detect_model([blob])[output_layer]
    license_score = []
    license_boxes = []
    data = res[0][0]
    index = 0
    for number, proposal in enumerate(data):
        if proposal[2] > 0.25:
            label = np.int32(proposal[1])
            confidence = proposal[2]
            xmin = np.int32(w * proposal[3])
            ymin = np.int32(h * proposal[4])
            xmax = np.int32(w * proposal[5])
            ymax = np.int32(h * proposal[6])

            if xmin < 0:
                xmin = 0
            if ymin < 0:
                ymin = 0
            if xmax >= w:
                xmax = w - 1
            if ymax >= h:
                ymax = h - 1
            vehicle_roi = frame[ymin:ymax, xmin:xmax,:]
            img = cv.resize(vehicle_roi, (72, 72))
            img = (np.float32(img) / 255.0 - 0.5) / 0.5
            img = img.transpose((2, 0, 1))
            x_input = torch.from_numpy(img).view(1, 3, 72, 72)
            color_, type_ = cnn_model(x_input.cuda())
            predict_color = torch.max(color_, 1)[1].cpu().detach().numpy()[0]
            predict_type = torch.max(type_, 1)[1].cpu().detach().numpy()[0]
            attrs_txt = "color:%s, type:%s"%(color_labels[predict_color], type_labels[predict_type])
            cv.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0, 0), 3)
            cv.putText(frame, attrs_txt, (xmin, ymin), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
    cv.imshow("Vehicle Attributes Recognition Demo", frame)
    res_key = cv.waitKey(1)
    if res_key == 27:
        break

cv.waitKey(0)
cv.destroyAllWindows()

