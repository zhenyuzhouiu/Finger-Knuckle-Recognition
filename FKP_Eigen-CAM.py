# EigenCam is gradients-free
import os
import torch
import cv2
import numpy as np
import torchvision.transforms as transforms
import warnings

from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, scale_cam_image
from PIL import Image
from models.net_model import ResidualFeatureNet, DeConvRFNet
from models.EfficientNetV2 import efficientnetv2_s, ConvBNAct\

warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

subject_path = r"dataset/PolyUKnuckleV3/yolov5/Session_1/1-104/5"
images_file = os.listdir(subject_path)

for i in images_file:
    image_path = os.path.join(subject_path, i)
    img = np.array(Image.open(image_path))
    img = cv2.resize(img, (128, 128))
    rgb_img = img.copy()
    img = np.float32(img) / 255
    transform = transforms.ToTensor()
    tensor = transform(img).unsqueeze(0)

    model = ResidualFeatureNet()
    model.load_state_dict(torch.load(r"checkpoint\RFNet\fkv3(yolov5)-session2_RFNet-wholeimagerotationandtranslation-lr0.001-subs8-angle5-a20-s4_2022-06-12-23-50\ckpt_epoch_1800.pth"))

    model.eval()
    model.cuda()
    target_layers = [model.conv5]

    cam = EigenCAM(model, target_layers, use_cuda=True)
    grayscale_cam = cam(tensor, aug_smooth=True, eigen_smooth=True)

    grayscale_cam = grayscale_cam[0]
    cam_image = show_cam_on_image(img, grayscale_cam, use_rgb=True)
    Image.fromarray(cam_image)

    result_img = Image.fromarray(np.hstack((rgb_img, cam_image)))
    result_img.show()
