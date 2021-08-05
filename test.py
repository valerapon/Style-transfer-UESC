import subprocess, os, argparse, shutil, sys
import numpy as np
import pandas as pd

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch import optim

from sklearn.neighbors import NearestNeighbors
from skimage.exposure import match_histograms

from skimage.feature import hog, local_binary_pattern
from PIL import Image
from VGG import *


CLASSES = [
        'Abstract_Expressionism', 'Action_painting', 'Analytical_Cubism', 'Art_Nouveau_Modern',
        'Baroque', 'Color_Field_Painting', 'Contemporary_Realism', 'Cubism',
        'Early_Renaissance', 'Expressionism', 'Fauvism', 'High_Renaissance',
        'Impressionism', 'Mannerism_Late_Renaissance', 'Minimalism', 'Naive_Art_Primitivism',
        'New_Realism', 'Northern_Renaissance', 'Pointillism', 'Pop_Art',
        'Post_Impressionism', 'Realism', 'Rococo', 'Romanticism',
        'Symbolism', 'Synthetic_Cubism', 'Ukiyo_e'
]
class_dict = {cls: i for i, cls in enumerate(CLASSES)}

class GramMatrix(nn.Module):
        def forward(self, input):
                b, c, h, w = input.size()
                F = input.view(b, c, h * w)
                G = torch.bmm(F, F.transpose(1, 2)) 
                G.div_(h * w)
                return G    
        
class GramMSELoss(nn.Module):
        def forward(self, input, target):
                out = nn.MSELoss()(GramMatrix()(input), target)
                return(out)

class StyleClassifier_HOG_LBP_VGG(nn.Module):
        def __init__(self):
                super(StyleClassifier_HOG_LBP_VGG, self).__init__()
                self.fc1 = nn.Linear(3545, 512)
                self.bn1  = nn.BatchNorm1d(512)              
                self.fc2 = nn.Linear(512, 128)
                self.dp2 = nn.Dropout(p=0.25) 
                self.fc3 = nn.Linear(128, 27)
                
        def forward(self, x):
                x = self.fc1(x)
                x = self.bn1(x)  
                x = self.fc2(x)
                x = self.dp2(x)
                x = self.fc3(x)
                return F.log_softmax(x, -1)
    
def resize_image(image):
        w, h = image.size
        if w > h:
                image = image.crop(((w - h) // 2, 0, (w - h) // 2 + h, h))
        elif w < h:
                image = image.crop((0, (h - w) // 2, w, (h - w) // 2 + w))
        return transforms.Resize(256)(image)

def get_vgg_vec(vgg, img):
        tensors = get_layers(vgg, img)
        vec = torch.cat([t.mean(axis=[0, 2, 3]) for t in tensors]).cpu()
        return vec

def get_ulbp_vec(img):    
        image = img.convert('LA')
        image = np.array(image)[:,:,0]
        lbp = local_binary_pattern(image, 8 * 3, 3, 'uniform')
        return np.histogram(lbp.ravel(), density=True, range=(0, 25), bins=25)[0]

def get_hog_vec(img):
        image = np.array(resize_image(img))
        fd = hog(image, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=False, multichannel=True)
        return fd

def load_data():
        print('LOAD DATA:', end=' ')
        database = np.load('Models/database_HOG_LBP_VGG_CONV.npy')
        img_name_list = pd.read_csv('Models/target.csv')['img']
        img_target = pd.read_csv('Models/target.csv')['target']
        print('OK')
        return database[:, 2073:3545], img_name_list, img_target

def load_VGG():
        print('LOAD VGG:', end=' ')
        vgg = VGG()
        vgg.load_state_dict(torch.load(os.getcwd() + '/Models/vgg_conv.pth'))
        for param in vgg.parameters():
                param.requires_grad = False
        if torch.cuda.is_available():
                vgg.cuda()
        print('OK')
        return vgg

def parse_args():
        parser = argparse.ArgumentParser(description='Testing function for style transfer using extended collection of styles')
        parser.add_argument('--content', type=str,  help='path to content image', required=True)
        parser.add_argument('--style',   nargs='+', help='paths to style images', required=True)
        return parser.parse_args()              

def prep(image, image_size=512):
        prep = transforms.Compose([transforms.Resize(image_size),
                                transforms.ToTensor(),
                                transforms.Lambda(lambda x: x[torch.LongTensor([2, 1, 0])]), #turn to BGR
                                transforms.Normalize(mean=[0.40760392, 0.45795686, 0.48501961], #subtract imagenet mean
                                                        std=[1, 1, 1]),
                                transforms.Lambda(lambda x: x.mul_(255)),
                                ])
        return prep(image)

def postp(tensor):
        postpa = transforms.Compose([transforms.Lambda(lambda x: x.mul_(1. / 255)),
                                transforms.Normalize(mean=[-0.40760392, -0.45795686, -0.48501961], #add imagenet mean
                                                        std=[1, 1, 1]),
                                transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])]), #turn to RGB
                                ])
        postpb = transforms.Compose([transforms.ToPILImage()])
        t = postpa(tensor)
        t[t > 1] = 1    
        t[t < 0] = 0
        img = postpb(t)
        return img

def style_transfer(content_path, style_path, same_style_list, out_path):
        model_dir = os.getcwd() + '/Models/'

        img_names = same_style_list + [content_path]
        imgs = [Image.open(name) for i, name in enumerate(img_names)]
        style = np.array(Image.open(style_path))
        for i in range(len(same_style_list)):
                imgs[i] = Image.fromarray(match_histograms(np.array(imgs[i]), style, multichannel=True))
        imgs_torch = [prep(img) for img in imgs]

        if torch.cuda.is_available():
                imgs_torch = [Variable(img.unsqueeze(0).cuda()) for img in imgs_torch]
        else:
                imgs_torch = [Variable(img.unsqueeze(0)) for img in imgs_torch]
        
        style_image, content_image = imgs_torch[:-1], imgs_torch[-1]

        
        opt_img = Variable(content_image.data.clone(), requires_grad=True)

        style_layers = ['r11', 'r21', 'r31', 'r41', 'r51'] 
        content_layers = ['r42']
        loss_layers = style_layers + content_layers
        loss_fns = [GramMSELoss()] * len(style_layers)  + [nn.MSELoss()] * len(content_layers)
        if torch.cuda.is_available():
                loss_fns = [loss_fn.cuda() for loss_fn in loss_fns]

        style_weights = [1e3 / n ** 2 for n in [64, 128, 256, 512, 512]]
        content_weights = [1e0]
        weights = style_weights + content_weights

        vgg = VGG()
        vgg.load_state_dict(torch.load(model_dir + 'vgg_conv.pth'))
        for param in vgg.parameters():
                param.requires_grad = False
        
        if torch.cuda.is_available():
                vgg.cuda()

        style_targets = []
        for img in style_image:
                tar = [GramMatrix()(A).detach() for A in vgg(img, style_layers)]
                if len(style_targets) == 0:
                        style_targets = tar
                else:
                        for i in range(len(tar)):
                                style_targets[i] += tar[i]
        for i in range(len(style_targets)):
                style_targets[i] /= float(len(style_image))
    
        content_targets = [A.detach() for A in vgg(content_image, content_layers)]
        targets = style_targets + content_targets

        optimizer = optim.LBFGS([opt_img])
        max_iter = 20

        for i in range(max_iter):
                def closure():
                        optimizer.zero_grad()
                        out = vgg(opt_img, loss_layers)
                        layer_losses = [weights[a] * loss_fns[a](A, targets[a]) for a, A in enumerate(out)]
                        loss = sum(layer_losses)
                        loss.backward()
                        return loss
                optimizer.step(closure)

        out_img = postp(opt_img.data[0].cpu().squeeze())
        out_img.save(out_path)
        torch.cuda.empty_cache()

def main(args):
        database, img_name_list, img_target = load_data()
        vgg = load_VGG() 

        model = torch.load('Models/model_HOG_LBP_VGG.pt').cuda()
        model.eval()

        for i, style in enumerate(args.style):
                print('%d: %s:' % (i + 1, style))
                img = Image.open(style)
                img = transforms.Resize(int(500 / max(img.size) * min(img.size)))(img)

                vec_hog = get_hog_vec(img)
                vec_lbp = get_ulbp_vec(img)
                vec_vgg = get_vgg_vec(vgg, img)

                vec = torch.tensor(np.concatenate([vec_hog, vec_lbp, vec_vgg[:-64]])[None,:], dtype=torch.float).cuda()

                res = model.forward(vec).argmax(axis=1).cpu()[0].item()

                nbrs = NearestNeighbors(n_neighbors=3, algorithm='ball_tree').fit(database[img_target == CLASSES[res]])
                index = nbrs.kneighbors(vec_vgg[:-64][None,:], return_distance=False)
                name_similar_image = img_name_list[img_target == CLASSES[res]].reset_index()['img'][index[0, 0]]

                print('\tclass:', CLASSES[res])
                print('\tsimilar image:', img_name_list[img_target == CLASSES[res]].reset_index()['img'][index[0, 0]])
                print('\tsimilar image:', img_name_list[img_target == CLASSES[res]].reset_index()['img'][index[0, 1]])
                print('\tsimilar image:', img_name_list[img_target == CLASSES[res]].reset_index()['img'][index[0, 2]])
                print('\tstylization:', end=' ')

                # subprocess.run(['style_transfer.py', args.content, './train/' + CLASSES[res] + '/' + name_similar_image, './output/' + str(i) + '.jpg'], shell=True)
                style_transfer(args.content, style, ['./train/' + CLASSES[res] + '/' + name_similar_image], './output/' + str(i) + '.jpg')
                shutil.copyfile('./train/' + CLASSES[res] + '/' + name_similar_image, './output/' + str(i) + '_style.jpg')
                print('./output/' + str(i) + '.jpg')
                print('\tstatus: OK')
        

if __name__ == '__main__':
        args = parse_args()
        main(args)