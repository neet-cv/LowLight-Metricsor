import os, re
import lpips
import filetype

import cv2 as cv
import numpy as np
from PIL import Image

from time import time
from NIQE import niqe
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim

import torch
import torchvision
import torch.nn as nn
from torchvision import transforms
import torchvision.models as models


class Baseline(nn.Module):
    def __init__(self):
        super(Baseline, self).__init__()
        self.backbone = torchvision.models.resnet50(pretrained=False)
        fc_feature = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(fc_feature, 1, bias=True)

    def forward(self, x):
        result = self.backbone(x)
        return result


class NIMA(nn.Module):
    """Neural IMage Assessment model by Google"""

    def __init__(self, base_model, num_classes=10):
        super(NIMA, self).__init__()
        self.features = base_model.features
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.75),
            nn.Linear(in_features=25088, out_features=num_classes),
            nn.Softmax())

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out


class Metrics:
    def __init__(self, mode: str, file_path: str, name: str = "Metrics", use_gpu: bool = True if torch.cuda.is_available() else False,
                 A_name: str = '_real_A', B_name: str = '_fake_B',
                 data_path: str = '',
                 A_path: str = '', B_path: str = ''):
        self.use_gpu = use_gpu
        self.dict = {
            'MAE': self.Compute_MAE,
            'MSE': self.Compute_MSE,
            'PSNR': self.Compute_PSNR,
            'SSIM': self.Compute_SSIM,
            'LPIPS': self.Compute_LPIPS,
            # 'LOE': self.Compute_LOE,
            'NIQE': self.Compute_NIQE,
            # 'SPAQ': self.Compute_SPAQ,
            'NIMA': self.Compute_NIMA
        }
        self.name = name
        self.result_paths = os.path.join(data_path)
        self.file_path = os.path.join(file_path)
        self.img_A_paths = []
        self.img_B_paths = []
        if mode == 'mixed':
            for root, _, file_names in sorted(os.walk(self.result_paths, followlinks=True)):
                for file_name in file_names:
                    path = os.path.join(root, file_name)
                    if A_name in file_name:
                        self.img_A_paths.append(path)
                    if B_name in file_name:
                        self.img_B_paths.append(path)
        elif mode == 'same_dir_parted':
            if os.path.isdir(data_path):
                list_name = sorted(os.listdir(data_path))
                for name in list_name:
                    if name == list_name[0]:
                        for file in os.listdir(os.path.join(data_path, name)):
                            path = os.path.join(data_path, name, file)
                            self.img_A_paths.append(path)
                    elif name == list_name[1]:
                        for file in os.listdir(os.path.join(data_path, name)):
                            path = os.path.join(data_path, name, file)
                            self.img_B_paths.append(path)
            else:
                print("Please input datafolders's parent directory! ")
        elif mode == 'diff_dir_parted':
            if os.path.isdir(A_path) and os.path.isdir(B_path):
                for file in os.listdir(A_path):
                    _path = os.path.join(A_path, file)
                    if filetype.is_image(_path):
                        self.img_A_paths.append(_path)
                for file in os.listdir(B_path):
                    _path = os.path.join(B_path, file)
                    if filetype.is_image(_path):
                        self.img_B_paths.append(_path)
        else:
            print(f"There isn't such mode named %s! " % mode)
        self.imgs_A = []
        self.imgs_B = []
        self.img_A_paths.sort(key=lambda f: [int(n) for n in re.findall(r"\d+", f)])
        self.img_B_paths.sort(key=lambda f: [int(n) for n in re.findall(r"\d+", f)])

        for img_A in self.img_A_paths:
            self.imgs_A.append(cv.imread(img_A))
        for img_B in self.img_B_paths:
            self.imgs_B.append(cv.imread(img_B))

        # Model init
        self.loss_fn = lpips.LPIPS(net='alex', pnet_rand=True, model_path="./models/alexnet-owt-7be5be79.pth")
        if self.use_gpu:
            self.loss_fn.cuda()

        model_pth = os.path.join(os.getcwd(), 'models/epoch-34.pth')
        pretrained_path = os.path.join(os.getcwd(), 'models/vgg16-397923af.pth')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.test_transform = transforms.Compose([
            transforms.Scale(256),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        base_model = models.vgg16(pretrained=False)
        base_model.load_state_dict(torch.load(pretrained_path, map_location=self.device))
        model = NIMA(base_model)
        try:
            model.load_state_dict(torch.load(model_pth, map_location=self.device))
            print('successfully loaded model')
        except IOError:
            print("Model doesn't exist! ")
            raise
        seed = 42
        torch.manual_seed(seed)
        self.model = model.to(self.device)
        self.model.eval()

    # To Reinitialize Images
    def __call__(self, mode: str, file_path: str,
                 A_name: str = '_real_A', B_name: str = '_fake_B',
                 data_path: str = '',
                 A_path: str = '', B_path: str = ''):
        self.img_A_paths = []
        self.img_B_paths = []
        self.imgs_A = []
        self.imgs_B = []
        if mode == 'mixed':
            for root, _, file_names in sorted(os.walk(self.result_paths, followlinks=True)):
                for file_name in file_names:
                    path = os.path.join(root, file_name)
                    if A_name in file_name:
                        self.img_A_paths.append(path)
                    if B_name in file_name:
                        self.img_B_paths.append(path)
        elif mode == 'same_dir_parted':
            if os.path.isdir(data_path):
                list_name = sorted(os.listdir(data_path))
                for name in list_name:
                    if name == list_name[0]:
                        for file in os.listdir(os.path.join(data_path, name)):
                            path = os.path.join(data_path, name, file)
                            self.img_A_paths.append(path)
                    elif name == list_name[1]:
                        for file in os.listdir(os.path.join(data_path, name)):
                            path = os.path.join(data_path, name, file)
                            self.img_B_paths.append(path)
            else:
                print("Please input datafolders's parent directory! ")
        elif mode == 'diff_dir_parted':
            if os.path.isdir(A_path) and os.path.isdir(B_path):
                for file in os.listdir(A_path):
                    _path = os.path.join(A_path, file)
                    if filetype.is_image(_path):
                        self.img_A_paths.append(_path)
                for file in os.listdir(B_path):
                    _path = os.path.join(B_path, file)
                    if filetype.is_image(_path):
                        self.img_B_paths.append(_path)
        else:
            print(f"There isn't such mode named %s! " % mode)
        self.img_A_paths.sort(key=lambda f: [int(n) for n in re.findall(r"\d+", f)])
        self.img_B_paths.sort(key=lambda f: [int(n) for n in re.findall(r"\d+", f)])
        for img_A in self.img_A_paths:
            self.imgs_A.append(cv.imread(img_A))
        for img_B in self.img_B_paths:
            self.imgs_B.append(cv.imread(img_B))
        self.result_paths = os.path.join(data_path)
        self.file_path = os.path.join(file_path)



    def __getitem__(self, item: str):
        i_th = len(os.listdir(self.file_path)) + 1
        if item.lower() == 'all':
            with open(os.path.join(self.file_path, 'Metrics_%d_th_%s.txt' % (i_th, self.name)), 'w') as fp:
                start_time = time()
                for metric in tqdm(self.dict.keys()):
                    fp.write(metric + ': ' + str(self.dict[metric.upper()]()) + '\n')
                end_time = time() - start_time
                fp.write("Total Time: " + str(end_time))
                return "Nothing"
        else:
            try:
                with open(os.path.join(self.file_path, 'Metrics.txt'), 'a') as fp:
                    res = self.dict[item.upper()]()
                    fp.write("Metrics_%d_th_experiment....\n" % i_th)
                    fp.write(item + ': ' + str(self.dict[item.upper()]()) + '\n')
                    fp.write("=" * 255)
                return item + ": " + str(res)
            except KeyError:
                print("The %s metric isn't included! " % item)

    def Compute_MAE(self):
        MAEs = []
        for img_A, img_B in zip(self.imgs_A, self.imgs_B):
            Error = np.abs(img_A - img_B)
            gray = cv.cvtColor(Error, cv.COLOR_BGR2GRAY)
            MAEs.append(np.mean(gray))
        return np.mean(MAEs)

    def Compute_MSE(self):
        MSEs = []
        for img_A, img_B in zip(self.imgs_A, self.imgs_B):
            img_A_gray = cv.cvtColor(img_A, cv.COLOR_BGR2GRAY)
            img_B_gray = cv.cvtColor(img_B, cv.COLOR_BGR2GRAY)
            MSEs.append((img_A_gray - img_B_gray) ** 2)
        return np.mean(MSEs)

    def Compute_PSNR(self):
        PSNRs = []
        for img_A, img_B in zip(self.imgs_A, self.imgs_B):
            PSNRs.append(cv.PSNR(img_A, img_B))
        return np.mean(PSNRs)

    def Compute_SSIM(self):
        SSIMs = []
        for img_A, img_B in zip(self.imgs_A, self.imgs_B):
            img_A_gray = cv.cvtColor(img_A, cv.COLOR_BGR2GRAY)
            img_B_gray = cv.cvtColor(img_B, cv.COLOR_BGR2GRAY)
            SSIMs.append(ssim(img_A_gray, img_B_gray))
        return np.mean(SSIMs)

    def Compute_LPIPS(self):
        LPIPSs = []
        for img_A_path, img_B_path in zip(self.img_A_paths, self.img_B_paths):
            img_A = lpips.im2tensor(lpips.load_image(img_A_path))
            img_B = lpips.im2tensor(lpips.load_image(img_B_path))
            if self.use_gpu:
                img_A = img_A.cuda()
                img_B = img_B.cuda()
            with torch.no_grad():
                LPIPSs.append(float(self.loss_fn(img_A, img_B)))
        return np.mean(LPIPSs)

    # def Compute_LOE(self):
    #     LOEs = []
    #     for img_A, img_B in zip(self.imgs_A, self.imgs_B):
    #         index_of_L_A = np.unravel_index(np.argmax(img_A), img_A.shape)
    #         index_of_L_B = np.unravel_index(np.argmax(img_B), img_B.shape)
    #         L_A = img_A[index_of_L_A]
    #         L_B = img_B[index_of_L_B]
    #         U_A = L_A >= img_A + 0
    #         U_B = L_B >= img_B + 0
    #         RD = U_A ^ U_B
    #         LOEs.append(np.mean(RD))
    #     return np.mean(LOEs)

    def Compute_NIQE(self):
        NIQEs = []
        for img_B_path in self.img_B_paths:
            img_B = np.array(Image.open(img_B_path).convert('LA'))[:, :, 0]
            NIQEs.append(niqe(img_B))
        return np.mean(NIQEs)

    # def Compute_SPAQ(self, size=512, input_size=224):
    #     SPAQs = []
    #     imgs_B = []
    #     for img_B_path in self.img_B_paths:
    #         img_B = Image.open(img_B_path)
    #         imgs_B.append(img_B)
    #     model = Baseline()
    #     for img_B in imgs_B:
    #         w_b, h_b = img_B.size
    #         if w_b >= size or h_b >= size:
    #             img_B = transforms.ToTensor()(transforms.Resize(size, Image.BILINEAR)(img_B))
    #         img_B = np.transpose(img_B, (2, 1, 0))
    #         img_shape_B = img_B.shape
    #         if len(img_shape_B) == 2:
    #             H_B, W_B, = img_shape_B
    #             num_of_channel_B = 1
    #         else:
    #             H_B, W_B, num_of_channel_B = img_shape_B
    #         if num_of_channel_B == 1:
    #             img_B = np.asarray([img_B, ] * 3, dtype=img_B.dtype)
    #
    #         stride = int(input_size / 2)
    #         hIdxMax_B = H_B - input_size
    #         wIdxMax_B = W_B - input_size
    #
    #         hIdx_B = [i * stride for i in range(int(hIdxMax_B / stride) + 1)]
    #         if H_B - input_size != hIdx_B[-1]:
    #             hIdx_B.append(H_B - input_size)
    #         wIdx_B = [i * stride for i in range(int(wIdxMax_B / stride) + 1)]
    #         if W_B - input_size != wIdx_B[-1]:
    #             wIdx_B.append(W_B - input_size)
    #         img_B = img_B.numpy()
    #         patches_numpy = [img_B[hId:hId + input_size, wId:wId + input_size, :]
    #                          for hId in hIdx_B
    #                          for wId in wIdx_B]
    #         patches_tensor = [transforms.ToTensor()(p) for p in patches_numpy]
    #         patches_tensor = torch.stack(patches_tensor, 0).contiguous()
    #         Image_B = patches_tensor.squeeze(0)
    #
    #         # if use_gpu:
    #         #     Image_B = Image_B.cuda()
    #         #     model = model.cuda()
    #         score_B = model(Image_B).mean()
    #         SPAQs.append(score_B.item())
    #     return np.mean(SPAQs)

    def Compute_NIMA(self):
        NIMAs = []
        imgs_B_BGR = []
        mean = 0.0
        # test_csv = os.path.join(ROOT_PATH, 'lib/test_labels.csv')
        for img_B in self.imgs_B:
            imgs_B_BGR.append(cv.cvtColor(img_B, cv.COLOR_BGR2RGB))
        for img_B in self.img_B_paths:
            imt_B = self.test_transform(Image.open(img_B))
            imt_B = imt_B.unsqueeze(dim=0)
            imt_B = imt_B.to(self.device)
            with torch.no_grad():
                out = self.model(imt_B)
            out = out.view(10, 1)
            for j, e in enumerate(out, 1):
                mean += j * e
            NIMAs.append(round(float(mean.cpu().numpy()), 3))
        return np.mean(NIMAs)

    def record_results(self, metric_name, metric_result):
        with open(os.path.join(self.file_path, 'Metrics.txt'), 'a') as fp:
            fp.write(metric_name + ': ' + str(metric_result) + '\n')
