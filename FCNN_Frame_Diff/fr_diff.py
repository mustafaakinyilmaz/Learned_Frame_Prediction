
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import glob
import os
import itertools
import warnings
warnings.filterwarnings('ignore')
import imageio
from natsort import natsorted
from skimage.measure import compare_ssim,compare_psnr
import logging
import time
import random
from model import m



logging.basicConfig(filename="fr_diff.log",level=logging.INFO)

device = torch.device("cuda:1")


def float_to_uint8(image):
    clip = np.clip(image,-1,1)
    original_range = np.round(((clip*127.5)+127.5))
    im_uint8 = np.uint8(original_range)
    return im_uint8


def video_list(path="/home/ml/Akin/Frame_Prediction/Dataset/Train/"):
    all_videos = natsorted(glob.glob(path+"*"))
    
    return all_videos


def train_data_loader(video_list):

    batch_im_list = []
    for video in video_list:
        im_list = natsorted(glob.glob(video+"/*"))
        batch_im_list.append(im_list)
    return batch_im_list



def test_data_loader(path="/home/ml/Akin/Frame_Prediction/Dataset/Test/"):
    folders = natsorted(glob.glob(path+"*"))
    for f in folders:
        im_list = natsorted(glob.glob(f+"/*"))
        yield(im_list)


def prepare_train_data(batch_im_list):

    X_train = []
    Y_train = []
    
    n_r = 96
    n_c = 96
    
    size = 5
    
    for train_im_list in batch_im_list:
    
        length = len(train_im_list)
 
        random_sq = np.random.permutation(length)
        diff = random_sq - size + 1
        random_start = diff[diff >= 0][0]
        split = train_im_list[random_start:random_start+size]
        
        reverse_select = [0,1]
        rand_reverse = random.sample(reverse_select,1)[0]
        
        if reverse_select == 1:
            split = split[::-1]

        x_train_array = []
        y_train_array = []
        
        rotate_select = [0,1,2,3] 
        rand_rotate = random.sample(rotate_select,1)[0]
        
        sample_im = imageio.imread(split[0])
        sample_im = np.rot90(sample_im,rand_rotate,axes=(0,1))
        (nw,nh) = sample_im.shape
        
        random_row_start = np.random.permutation(nw) - n_r
        random_row_start = random_row_start[random_row_start >= 0][0]
        
        random_col_start = np.random.permutation(nh) - n_c
        random_col_start = random_col_start[random_col_start >= 0][0]
        
        for k in range(size-2):

            x1_image = imageio.imread(split[k]).astype(np.float32)
            x2_image = imageio.imread(split[k+1]).astype(np.float32)
            
            diff = (x2_image - x1_image)/255.0
            
            diff_rotated = np.rot90(diff,rand_rotate,axes=(0,1))
            x_train_array.append(diff_rotated[random_row_start:random_row_start+n_r,random_col_start:random_col_start+n_c])

        y_diff = (imageio.imread(split[k+2]).astype(np.float32) - imageio.imread(split[k+1]).astype(np.float32))/255.0
        y_diff_rotated = np.rot90(y_diff,rand_rotate,axes=(0,1))
        y_train_array.append(y_diff_rotated[random_row_start:random_row_start+n_r,random_col_start:random_col_start+n_c])
        
        X_train.append(x_train_array)
        Y_train.append(y_train_array)

    X_train = np.array(X_train)
    Y_train = np.array(Y_train)

    return X_train,Y_train


def prepare_test_data(test_im_list):
    X_test = []
    Y_test = []
    last_Y = []
    real_future = []
    
    size = 4
    
    for k in range(len(test_im_list)-size):
        x_test_array = []
        for i in range(size-1):
            
            x_diff = (imageio.imread(test_im_list[k+i+1]).astype(np.float32) - imageio.imread(test_im_list[k+i]).astype(np.float32))/255.0
            (nw,nh) = x_diff.shape
            x_test_array.append(x_diff)
        
        X_test.append(x_test_array)
        y_diff = (imageio.imread(test_im_list[k+i+2]).astype(np.float32) - imageio.imread(test_im_list[k+i+1]).astype(np.float32))/255.0
        Y_test.append(y_diff.reshape(1,nw,nh))
        
        last_Y.append(imageio.imread(test_im_list[k+i+1]).astype(np.float32).reshape(1,nw,nh))
        real_future.append(imageio.imread(test_im_list[k+i+2]).reshape(1,nw,nh))
    
    return np.array(X_test),np.array(Y_test),np.array(last_Y),np.array(real_future)


def calculate_loss(out,real):
    loss = torch.mean((out-real)**2)
    return loss


def train_one_step(train_ims,model,optimizer):
    model = model.train()

    X_train,Y_train = prepare_train_data(train_ims)
    
    inp,real = torch.from_numpy(X_train).to(device).float(),torch.from_numpy(Y_train).to(device).float()
            
    out = model.forward(inp)
    loss = calculate_loss(out,real)
                
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
        

    return loss.item()





def test_and_save(model):
    with torch.no_grad():
        
        model.eval()

        total_test_loss = 0
        total_test_psnr = 0
        total_test_ssim = 0

        test_im_batch = test_data_loader()

        num_of_test_videos = 0
        folder_names = ["coastguard","container","football","foreman","garden","hall_monitor","mobile","tennis"]
        folder = 0

        total_frames = 0

        time_start = time.time()
        for test_ims in test_im_batch:
            X_test,Y_test,last_Y_test,real_future = prepare_test_data(test_ims)
            
            (m,nc,nw,nh) = Y_test.shape

            video_psnr = 0
            video_ssim = 0

            d = 4
            os.makedirs("Created/"+folder_names[folder],exist_ok=True)

            for frame in range(m):
                inp = torch.from_numpy(X_test[frame:frame+1]).to(device).float()
                real = torch.from_numpy(Y_test[frame:frame+1]).to(device).float()

                out = model.forward(inp)
                
                reconstructed = np.round(np.clip((out.cpu().detach().numpy()[0,0]*255 + last_Y_test[frame,0]),0,255)).astype(np.uint8)
                ground_truth = np.round(real_future[frame,0]).astype(np.uint8)

                psnr = compare_psnr(ground_truth,reconstructed,data_range=255)
                video_psnr += psnr
                ssim = compare_ssim(ground_truth,reconstructed,data_range=255)
                video_ssim += ssim

                save_path = "Created/"+folder_names[folder]+"/frame"+str(d)+".png"
                imageio.imsave(save_path,reconstructed)
                d += 1
                total_frames += 1

            num_of_test_videos += 1

            average_video_psnr = video_psnr/X_test.shape[0]
            total_test_psnr += average_video_psnr

            average_video_ssim = video_ssim/X_test.shape[0]
            total_test_ssim += average_video_ssim

            folder += 1
        time_end = time.time()

        average_test_psnr = total_test_psnr / num_of_test_videos
        average_test_ssim = total_test_ssim / num_of_test_videos

        test_time = time_end - time_start

        fps = total_frames / test_time
    
    return average_test_psnr,average_test_ssim,fps



def save_model(model,optimizer):
    state = {
        "state_dict" : model.state_dict(),
        "optimizer" : optimizer.state_dict()
    }
    torch.save(state,"fr_diff.pth")


# In[ ]:


def main():
    
    np.random.seed(1)
    torch.manual_seed(2)
    random.seed(3)
    
    num_input_diff = 3
    filter_size = 64
    conv_kernel = 3
    num_of_rdb = 10
    
    total_train_step = 600000
    train_step = 2000
    lr_step = 100000
    
    learning_rate = 1.e-4
            
    model = m.Model(num_input_diff,filter_size,conv_kernel,num_of_rdb)
    
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    logging.info("number of parameters: "+str(params))

    model = model.to(device).float()
    

    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)

    average_train_loss = 0    

    best_test_psnr = 0

    all_videos = video_list()
    
    logging.info("train video samples: "+str(len(all_videos)))
    logging.info("**********")

    batch_size = 8
    
    time_start = time.time()
    for minibatch_processed in range(1,total_train_step+1):
        
        batch_video = random.sample(all_videos,batch_size)
        train_im_list = train_data_loader(batch_video)
        train_step_loss = train_one_step(train_im_list,model,optimizer)
        average_train_loss += train_step_loss
        
                                
        if  minibatch_processed % train_step == 0:
            average_test_psnr,average_test_ssim,test_fps = test_and_save(model)
            if average_test_psnr > best_test_psnr:
                best_test_psnr = average_test_psnr
                logging.info("NEW BEST !!!")
                
                save_model(model,optimizer)
            
            logging.info("number of iterations: "+str(minibatch_processed))
            logging.info("learning rate: "+str(learning_rate))
            logging.info("train_loss: "+str(average_train_loss/train_step))
            logging.info("-----")
            logging.info("fps: "+str(test_fps))
            logging.info("test ssim: "+str(average_test_ssim))
            logging.info("test psnr: "+str(average_test_psnr))
            logging.info("best psnr: "+str(best_test_psnr))
            
            logging.info("****************")

            average_train_loss = 0

        if minibatch_processed % lr_step == 0:
            learning_rate /= 2
            for g in optimizer.param_groups:
                g["lr"] = learning_rate
            
            
        
    time_end = time.time()
    training_time = time_end-time_start
    day = training_time // (24 * 3600)
    training_time = training_time % (24 * 3600)
    hour = training_time // 3600
    training_time %= 3600
    minutes = training_time // 60
    training_time %= 60
    seconds = training_time
    logging.info("day:hour:minute:second-> %d:%d:%d:%d" % (day, hour, minutes, seconds))


# In[ ]:


main()

