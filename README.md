# RetinaFace in OpenVINO

This is OpenVINO based webcam demo for RetinaFace (Simultaneous face and landmark detection).
Forked from PyTorch implementation that is available here: [https://github.com/wang-xinyu/Pytorch_Retinaface](https://github.com/wang-xinyu/Pytorch_Retinaface).

Torch dependency for demo has been eliminated and code can run without it. Some postprocessing steps are kept as is with minimal optimizations performed.

## Steps to reproduce

### Clone source and install packages
```Shell
git clone https://github.com/yury-gorbachev/OpenVINO_Retinaface
conda create --name ov_retinaface python=3.7
conda activate ov_retinaface
cd OpenVINO_Retinaface
pip install -r requirements.txt
```

### Install OpenVINO and development tools
```Shell
pip install openvino-dev >= 22.1
```

### Download pretrained models as per original repo instructions

Pretrain model  and trained model are put in [google cloud](https://drive.google.com/open?id=1oZRSG0ZegbVkVwUd8wUIQx8W7yfZ_ki1) and [baidu cloud](https://pan.baidu.com/s/12h97Fy1RYuqMMIV-RpzdPg) Password: fstq . The model could be put as follows:
```Shell
  ./weights/
      mobilenet0.25_Final.pth
      mobilenetV1X0.25_pretrain.tar
      Resnet50_Final.pth
```

### Convert model to onnx (Will generate MobileNet baset model by default)

```Shell
python ./convert_to_onnx.py
```
This will generate FaceDetector.onnx

### Convert model to OpenVINO IR (Will generate MobileNet baset model by default)

```Shell
mo --use_new_frontend --mean_values="[104, 117, 123]" --input_model=FaceDetector.onnx --model_name=FaceDetector_MN
```
This will generate FaceDetector_MN.xml and *.bin files that represent model in OV IR.
Mean values are integrated in model graph, no need for additional preprocessing.

### Run demo (requries camera)
```Shell
python openvino_webcam_demo.py
```
Demo will run on CPU by default with first available video capture device.

## Trying with different model

### Using ResNet-50 backbone

Generate onnx file that corresponds to model with RN-50.

```Shell
python ./convert_to_onnx.py -m ./weights/Resnet50_Final.pth --network resnet50
```

### Convert model to OpenVINO IR 

```Shell
mo --use_new_frontend --mean_values="[104, 117, 123]" --input_model=FaceDetector.onnx --model_name=FaceDetector_RN50
```
This will generate FaceDetector_RN50.xml and *.bin files that represent model in OV IR.
Mean values are integrated in model graph, no need for additional preprocessing.

### Run demo (requries camera)
```Shell
python openvino_webcam_demo.py --backbone=RN
```

### Using different inference device

Use --device option to change device that will be used for inference (e.g. GPU). List of available devices is printed upon demo run.
Check out [openvino.ai](openvino.ai) for information on configuration

```Shell
python openvino_webcam_demo.py --device=GPU
```
