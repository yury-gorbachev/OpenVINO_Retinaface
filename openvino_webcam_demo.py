#!/usr/bin/env python

import cv2
import numpy as np
import time
import argparse

from math import ceil
from itertools import product as product

from openvino.runtime import Core, Layout, Type
from openvino.preprocess import PrePostProcessor, ResizeAlgorithm

from data import cfg_mnet, cfg_re50
from utils.nms.py_cpu_nms import py_cpu_nms

#model postprocessing (modified from original pytorch based code)
def compute_priors(cfg, image_size):
    min_sizes_cfg = cfg['min_sizes']
    steps = cfg['steps']
    clip = cfg['clip']
    feature_maps = [[ceil(image_size[0]/step), ceil(image_size[1]/step)] for step in steps]
    name = "s"
    anchors = []
    for k, f in enumerate(feature_maps):
        min_sizes = min_sizes_cfg[k]
        for i, j in product(range(f[0]), range(f[1])):
            for min_size in min_sizes:
                s_kx = min_size / image_size[1]
                s_ky = min_size / image_size[0]
                dense_cx = [x * steps[k] / image_size[1] for x in [j + 0.5]]
                dense_cy = [y * steps[k] / image_size[0] for y in [i + 0.5]]
                for cy, cx in product(dense_cy, dense_cx):
                        anchors += [cx, cy, s_kx, s_ky]

    # back to torch land
    output =np.asarray(anchors).reshape(-1,4)
    if clip:
        output=np.clip(output, 0, 1)
    return output


def decode_boxes(loc, priors, variances):
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc (tensor): location predictions for loc layers,
            Shape: [num_priors,4]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded bounding box predictions
    """

    boxes = np.concatenate((
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:], 
        priors[:, 2:] * np.exp(loc[:, 2:] * variances[1])),
        axis=1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]

    return boxes

def decode_landmarks(pre, priors, variances):
    """Decode landm from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        pre (tensor): landm predictions for loc layers,
            Shape: [num_priors,10]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded landm predictions
    """
    landms = np.concatenate((priors[:, :2] + pre[:, :2] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 2:4] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 4:6] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 6:8] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 8:10] * variances[0] * priors[:, 2:],
                        ), axis=1)
    return landms

def postprocess_frame(cfg, output):

    confidence=output[1]
    scores = confidence.squeeze(0)[:, 1]
    locations=output[0].squeeze(0)
    landmarks=output[2].squeeze(0)

    # ignore low scores
    indexes = np.where(scores > confidence_threshold)[0]
    filtered_priors=priors[indexes]
    filtered_locations=locations[indexes]
    filtered_landmarks = landmarks[indexes]
    scores = scores[indexes]

    boxes = decode_boxes(filtered_locations, filtered_priors, cfg['variance'])
    #to original image size
    boxes = boxes * np.float32([input_w, input_h, input_w, input_h]) 

    landms = decode_landmarks(filtered_landmarks, filtered_priors, cfg['variance'])
    #to original image size
    landms = landms * np.float32([input_w, input_h, input_w, input_h,
                            input_w, input_h, input_w, input_h,
                            input_w, input_h])

    # keep top-K before NMS
    order = scores.argsort()[::-1][:top_k]
    boxes = boxes[order.copy()]
    landms = landms[order.copy()]
    scores = scores[order.copy()]

    # do NMS
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = py_cpu_nms(dets, nms_threshold)
    dets = dets[keep, :]
    landms = landms[keep]

    # keep top-K faster NMS
    dets = dets[:keep_top_k, :]
    landms = landms[:keep_top_k, :]

    dets = np.concatenate((dets, landms), axis=1)
    return dets

#actual processing of frame
def process_frame(model, input):
    input=np.expand_dims(input, 0) #HWC->NHWC
    tic=time.time()
    output=model.infer_new_request({0: input})
    infer_time=(time.time() - tic)
    output=list(output.values())
    output=(output[0], output[1], output[2])
    return output, infer_time


parser = argparse.ArgumentParser(description='OV_RF')
parser.add_argument('--device', default="CPU", help='Device to perform inference on')
parser.add_argument('--backbone', default="MN", help='Model to run (MN/RN50). Requires IRs to present before run')

args = parser.parse_args()

#algo settings

if (args.backbone=="MN"):
    #for mobilenet
    print("Demo will run MobineNet based model")
    model_cfg=cfg_mnet
    modelPath="FaceDetector_MN.xml"
else:
    #for resnet-50
    print("Demo will run ResNet based model")
    modelPath="FaceDetector_RN50.xml"
    model_cfg=cfg_re50

confidence_threshold=0.02
top_k=int(5)
nms_threshold=0.4
keep_top_k=750
vis_threshold=0.6

#figure out actual size of image that we will be getting from camera
camera = cv2.VideoCapture(0)
ret, frame=camera.read()
input_h, input_w, _ = frame.shape

#create core object
core=Core()

print("Devices available for OpenVINO inference:")
for device in core.available_devices:
    print(f'- {device}')

print("Demo will run on "+args.device)

#load IR model, converted from ONNX and mean values integrated in model
model=core.read_model(modelPath)

#this is to learn which input size model expects
_, _, model_h, model_w = model.inputs[0].get_shape()
scale_h=input_h/model_h
scale_w=input_w/model_w
priors = compute_priors(model_cfg, (model_h, model_w))


#preprocessing that will be integrated into compiled model
#executed for every inference call on target device
pp=PrePostProcessor(model)
#input tensor corresponds to what we have got from camera
pp.input().tensor() \
    .set_element_type(Type.u8) \
    .set_layout(Layout('NHWC')) \
    .set_spatial_static_shape(input_h, input_w)
#convert input tensor from u8->fp32
pp.input().preprocess().convert_element_type(Type.f32)
#resize to model input size
pp.input().preprocess().resize(ResizeAlgorithm.RESIZE_LINEAR)
#this will convert layout from NHWC to NCHW 
pp.input().model().set_layout(Layout('NCHW'))
#integrate preprocessing into model
model=pp.build()

#compile model for CPU
compiledModel=core.compile_model(model, args.device)

while True:
    ret, frame = camera.read()
    
    if not ret:
        print("capture failure")
        break
        
    #just pass frame unchanged - preprocessing will happen inside infer call
    output, infer_time=process_frame(compiledModel, frame)
    dets=postprocess_frame(model_cfg, output)
    infer_time='OpenVINO Inference time: '+'{:.0f}ms'.format(infer_time*1000)+' ({:.1f} FPS)'.format(1/infer_time)
 
    #draw detections and landmarks
    for b in dets:
        if (b[4]<vis_threshold):
            continue
        b = list(map(int, b))
        cv2.rectangle(frame, (int(b[0]), int(b[1])), (int(b[2]),int(b[3])), (0, 0, 255), 1)
        
        cv2.circle(frame, (int(b[5]), int(b[6])), 1, (0, 0, 255), 4)
        cv2.circle(frame, (int(b[7]), int(b[8])), 1, (0, 255, 255), 4)
        cv2.circle(frame, (int(b[9]), int(b[10])), 1, (255, 0, 255), 4)
        cv2.circle(frame, (int(b[11]), int(b[12])), 1, (0, 255, 0), 4)
        cv2.circle(frame, (int(b[13]), int(b[14])), 1, (255, 0, 0), 4)

    cv2.putText(frame, infer_time, (0, 20),
                            cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

    cv2.imshow("output", frame)

    key = cv2.waitKey(1)
    if key%256==27:
        break

camera.release()

cv2.destroyAllWindows()
