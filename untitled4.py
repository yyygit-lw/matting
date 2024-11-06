# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 16:28:40 2023

@author: BangweiYe
"""

# -*- coding: utf-8 -*-
"""
Created on Mon May 15 20:04:37 2023

@author: 67413
"""
import os
import numpy as np
import cv2
from sklearn.ensemble import RandomForestClassifier
from skimage import feature as ft
from skimage.measure import label
import scipy
import matplotlib.pyplot as plt
import collections
import random
from joblib import Parallel, delayed
from time import *
MAX_INT32=2147483640
MIN_F=1E-7

def show(img, channel=1):
    if channel == 3:
        plt.imshow(img)
    elif channel == 1:
        plt.imshow(img, cmap='gray')
    else:
        return
    plt.show()


def error_distribution(Alpha,Trimap,alpha_pred,title,save_path):
    rows, cols = Alpha.shape
    error=[Alpha[i,j]-alpha_pred[i,j] for i in range(rows) for j in range(cols) if Trimap[i,j]==128]

    plt.figure() #初始化一张图
    plt.hist(error,bins=51)  #直方图关键操作
    plt.grid(alpha=0.5,linestyle='-.') #网格线，更好看 
    plt.xlabel('error')
    plt.ylabel('Number of error')
    plt.title(Img_file+":"+title)
    if save_path!=-1:
        error_filename=Img_file[:-4]+"_error分布图.png"
        plt.savefig(os.path.join(save_path+"error_distribution\\"+title,error_filename),dpi=500,bbox_inches = 'tight')   # 保存图片 注意 在show()之前  不然show会重新创建新的 图片   
    plt.show()
    
def visual_result(Img,Trimap_dir,prior,alpha_samp,alpha_weight,Alpha,save_path,Img_file):
    ## 依次排列：Img,Trimap,prior,alpha_samp,alpha_weight,Alpha
    rows, cols = Alpha.shape
    Img2 = Img[:,:,::-1]
    plt.figure(1)
    plt.subplot(1, 6, 1)
    plt.imshow(Img2)
    plt.xticks([]),plt.yticks([])
    
    plt.figure(1)
    plt.subplot(1,6, 2)
    Trimap_real = cv2.imread(Trimap_dir, cv2.IMREAD_GRAYSCALE)
    plt.imshow(Trimap_real, cmap='gray')
    plt.xticks([]),plt.yticks([])

    
    plt.figure(1)
    plt.subplot(1, 6, 3)
    plt.imshow(prior, cmap='gray')
    plt.xticks([]),plt.yticks([])
    
    plt.figure(1)
    plt.subplot(1, 6, 4)
    plt.imshow(alpha_samp, cmap='gray')
    plt.xticks([]),plt.yticks([])
    
    plt.figure(1)
    plt.subplot(1, 6, 5)
    plt.imshow(alpha_weight, cmap='gray')
    plt.xticks([]),plt.yticks([])
    
    plt.figure(1)
    plt.subplot(1, 6, 6)
    plt.imshow(Alpha, cmap='gray')
    plt.xticks([]),plt.yticks([])
    if save_path!=-1:
        save=os.path.join(save_path+"result",Img_file[:-4]+"_.png")
        plt.savefig( save,dpi=500,bbox_inches = 'tight')
        
    error_distribution(Alpha,Trimap_real,prior,"prior",save_path)
    error_distribution(Alpha,Trimap_real,alpha_samp,"alpha_samp",save_path)
    error_distribution(Alpha,Trimap_real,alpha_weight,"alpha_weight",save_path)


def show(img, channel=1):
    if channel == 3:
        plt.imshow(img)
    elif channel == 1:
        plt.imshow(img, cmap='gray')
    else:
        return
    plt.show()

def compute_mse_loss(pred, target, trimap=None):
	"""
	% pred: the predicted alpha matte
	% target: the ground truth alpha matte
	% trimap: the given trimap
	"""
	error_map = np.array(pred.astype('int32')-target.astype('int32'))/255.0
	if trimap is not None:
		loss = sum(sum(error_map**2 * (trimap == 128))) / sum(sum(trimap == 128))
	else:
		h, w = pred.shape
		loss = sum(sum(error_map ** 2)) / (h*w)
	return loss

def compute_sad_loss(pred, target, trimap=None):
	"""
	% the loss is scaled by 1000 due to the large images used in our experiment.
	% Please check the result table in our paper to make sure the result is correct.
	"""
	error_map = np.abs(pred.astype('int32')-target.astype('int32'))/255.0
	if trimap is not None:
		loss = np.sum(np.sum(error_map*(trimap == 128)))
	else:
		loss = np.sum(np.sum(error_map))
	loss = loss / 1000
	return loss

def getLargestCC(segmentation):
    labels = label(segmentation, connectivity=1)
    largestCC = labels == np.argmax(np.bincount(labels.flat))
    return largestCC

def compute_connectivity_error(pred, target, trimap):
    step=0.1
    pred = pred / 255.0
    target = target / 255.0
    h, w = pred.shape

    thresh_steps = list(np.arange(0, 1 + step, step))
    l_map = np.ones_like(pred, dtype=np.float) * -1
    for i in range(1, len(thresh_steps)):
        pred_alpha_thresh = (pred >= thresh_steps[i]).astype(np.int)
        target_alpha_thresh = (target >= thresh_steps[i]).astype(np.int)

        omega = getLargestCC(pred_alpha_thresh * target_alpha_thresh).astype(np.int)
        flag = ((l_map == -1) & (omega == 0)).astype(np.int)
        l_map[flag == 1] = thresh_steps[i - 1]

    l_map[l_map == -1] = 1

    pred_d = pred - l_map
    target_d = target - l_map
    pred_phi = 1 - pred_d * (pred_d >= 0.15).astype(np.int)
    target_phi = 1 - target_d * (target_d >= 0.15).astype(np.int)
    loss = np.sum(np.abs(pred_phi - target_phi)[trimap == 128])

    return loss

def gauss(x, sigma):
    y = np.exp(-x ** 2 / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))
    return y

def dgauss(x, sigma):
    y = -x * gauss(x, sigma) / (sigma ** 2)
    return y

def gaussgradient(im, sigma):
    epsilon = 1e-2
    halfsize = np.ceil(sigma * np.sqrt(-2 * np.log(np.sqrt(2 * np.pi) * sigma * epsilon))).astype(np.int32)
    size = 2 * halfsize + 1
    hx = np.zeros((size, size))
    for i in range(0, size):
        for j in range(0, size):
            u = [i - halfsize, j - halfsize]
            hx[i, j] = gauss(u[0], sigma) * dgauss(u[1], sigma)

    hx = hx / np.sqrt(np.sum(np.abs(hx) * np.abs(hx)))
    hy = hx.transpose()

    gx = scipy.ndimage.convolve(im, hx, mode='nearest')
    gy = scipy.ndimage.convolve(im, hy, mode='nearest')

    return gx, gy

def compute_gradient_loss(pred, target, trimap):

    pred = pred / 255.0
    target = target / 255.0

    pred_x, pred_y = gaussgradient(pred, 1.4)
    target_x, target_y = gaussgradient(target, 1.4)

    pred_amp = np.sqrt(pred_x ** 2 + pred_y ** 2)
    target_amp = np.sqrt(target_x ** 2 + target_y ** 2)

    error_map = (pred_amp - target_amp) ** 2
    loss = np.sum(error_map[trimap == 128])

    return loss


def get_lbp(src):
    #param src:灰度图像
    #return:lbp特征
    height, width = src.shape[:2]
    dst = np.zeros((height, width), dtype=np.uint8)

    lbp_value = np.zeros((1,8), dtype=np.uint8)
    neighbours = np.zeros((1,8), dtype=np.uint8)
    for row in range(1, height-1):
        for col in range(1, width-1):
            center = src[row, col]

            neighbours[0, 0] = src[row-1, col-1]
            neighbours[0, 1] = src[row-1, col]
            neighbours[0, 2] = src[row-1, col+1]
            neighbours[0, 3] = src[row, col+1]
            neighbours[0, 4] = src[row+1, col+1]
            neighbours[0, 5] = src[row+1, col]
            neighbours[0, 6] = src[row+1, col-1]
            neighbours[0, 7] = src[row, col-1]

            for i in range(8):
                if neighbours[0, i] > center:
                    lbp_value[0, i] = 1
                else:
                    lbp_value[0, i] = 0

            # 转成二进制数
            lbp = lbp_value[0, 0] * 1 + lbp_value[0, 1] * 2 + lbp_value[0, 2] * 4 + lbp_value[0, 3] * 8 \
                + lbp_value[0, 4] * 16 + lbp_value[0, 5] * 32 + lbp_value[0, 6] * 64 + lbp_value[0, 7] * 128
            dst[row, col] = lbp
    return dst

def lbp_uniform(img):
    def cal_basic_lbp(img,i,j):#比中心像素大的点赋值为1，比中心像素小的赋值为0，返回得到的二进制序列
        sum = []
        if img[i - 1, j ] > img[i, j]:
            sum.append(1)
        else:
            sum.append(0)
        if img[i - 1, j+1 ] > img[i, j]:
            sum.append(1)
        else:
            sum.append(0)
        if img[i , j + 1] > img[i, j]:
            sum.append(1)
        else:
            sum.append(0)
        if img[i + 1, j+1 ] > img[i, j]:
            sum.append(1)
        else:
            sum.append(0)
        if img[i + 1, j ] > img[i, j]:
            sum.append(1)
        else:
            sum.append(0)
        if img[i + 1, j - 1] > img[i, j]:
            sum.append(1)
        else:
            sum.append(0)
        if img[i , j - 1] > img[i, j]:
            sum.append(1)
        else:
            sum.append(0)
        if img[i - 1, j - 1] > img[i, j]:
            sum.append(1)
        else:
            sum.append(0)
        return sum   
    def bin_to_decimal(bin):#二进制转十进制
        res = 0
        bit_num = 0 #左移位数
        for i in bin[::-1]:
            res += i << bit_num   # 左移n位相当于乘以2的n次方
            bit_num += 1
        return res    
    def calc_sum(r):  # 获取值r的二进制中跳变次数
        sum_ = 0
        for i in range(0,len(r)-1):
            if(r[i] != r[i+1]):
                sum_ += 1
        return sum_    
    
    uniform_map = {
    0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 6: 5, 7: 6, 8: 7, 12: 8,14: 9, 15: 10, 16: 11, 24: 12, 28: 13, 30: 14, 31: 15, 32: 16, 48: 17,
 56: 18, 60: 19, 62: 20, 63: 21, 64: 22, 96: 23, 112: 24,120: 25, 124: 26, 126: 27, 127: 28, 128: 29, 129: 30, 131: 31, 135: 32,143: 33,
 159: 34, 191: 35, 192: 36, 193: 37, 195: 38, 199: 39, 207: 40,223: 41, 224: 42, 225: 43, 227: 44, 231: 45, 239: 46, 240: 47, 241: 48,
243: 49, 247: 50, 248: 51, 249: 52, 251: 53, 252: 54, 253: 55, 254: 56,255: 57}    
    revolve_array = np.zeros(img.shape,np.uint8)
    width = img.shape[0]
    height = img.shape[1]
    for i in range(1,width-1):
        for j in range(1,height-1):
            sum_ = cal_basic_lbp(img,i,j) #获得二进制
            num_ = calc_sum(sum_)  #获得跳变次数
            if num_ <= 2:
                revolve_array[i,j] = uniform_map[bin_to_decimal(sum_)] #若跳变次数小于等于2，则将该二进制序列对应的十进制值就是邻域中心的LBP值，因为只有58种可能的值，但值得最大值可以是255，所以这里进行映射。
            else:
                revolve_array[i,j] = 58
    return revolve_array

def get_Imgfeat(Img):
    Img_gray=cv2.cvtColor(Img,cv2.COLOR_BGR2GRAY)
    Img_grad_x,Img_grad_y = cv2.Sobel(Img_gray, cv2.CV_32F, 1, 0),cv2.Sobel(Img_gray, cv2.CV_32F, 0, 1)
    Img_gradx, Img_grady = cv2.convertScaleAbs(Img_grad_x), cv2.convertScaleAbs(Img_grad_y)   
    Img_grad = cv2.addWeighted(Img_gradx,0.5, Img_grady, 0.5, 0)

    
    gaussianBlur = cv2.GaussianBlur(Img_gray, (3,3), 0)
    _, binary = cv2.threshold(gaussianBlur, 127, 255, cv2.THRESH_BINARY)
    dst = cv2.Laplacian(binary, cv2.CV_16S, ksize = 3)
    Img_lap = cv2.convertScaleAbs(dst)
    
    Img_lbp_unfrevolve= lbp_uniform(Img_gray)
    Img_gray=Img_gray.astype(int)
   
    Img_lbp=np.array(Img_gray,dtype='uint8')
    Img_lbp=get_lbp(Img_lbp)
    Img_lbp=Img_lbp.reshape(rows,cols,1)
    '''
    show(Img_var)
    '''
    Img_id=np.zeros((rows,cols,2))

    for i in range(rows):
        for j in range(cols):
            Img_id[i,j][0]=i
            Img_id[i,j][1]=j
    Img_hsv = cv2.cvtColor(Img, cv2.COLOR_BGR2HSV)
    Img_lab = cv2.cvtColor(Img, cv2.COLOR_BGR2LAB)
    
    '''
    Img_var=np.zeros((rows,cols))
    for i in range(rows):
        for j in range(cols):
            Nb_color=[Img[m][n] for (m,n) in [(i-1,j-1),(i-1,j),(i-1,j+1),(i,j-1),(i,j+1),(i+1,j-1),(i+1,j),(i+1,j+1)]
                      if 0<=m<rows and 0<=n<cols]
            Img_var[i][j]=np.var(Nb_color)  
    
    Img_var=Img_var.reshape(rows,cols,1)
    '''
    from skimage import  exposure
    Img_hog = ft.hog(Img_gray,orientations=9,pixels_per_cell=[16,16],cells_per_block=[2,2],visualize=True)[1]
    Img_hog= exposure.rescale_intensity(Img_hog, in_range=(0, 10))
    Img_hog=Img_hog.reshape(rows,cols,1)
    Img_gray=Img_gray.reshape(rows,cols,1)
    Img_grad=Img_grad.reshape(rows,cols,1)
    Img_lap=Img_lap.reshape(rows,cols,1)
    Img_lbp_unfrevolve=Img_lbp_unfrevolve.reshape(rows,cols,1)
    return  np.concatenate((Img_lab,Img_hsv,Img_id,Img_lap,Img_lbp,Img_lbp_unfrevolve,Img_grad,Img_hog),axis=2)#np.concatenate((Img_lab,Img_hsv,Img_lbp,Img_lbp_unfrevolve,Img_lap,Img_grad,Img_hog),axis=2)#np.concatenate((Img_gray,Img_lab,Img_hsv,Img_id,Img_lbp,Img_var,lab_grad,Img_hog),axis=2)

def get_estimate_alpha(i, f, b): #i,f,b必须为int
    tmp1,tmp2 = i-b,f-b
    tmp3,tmp4 = sum(tmp1 * tmp2),sum(tmp2 * tmp2)
    if tmp4 == 0:
        return 1
    else:
        return min(max(tmp3 / tmp4, 0.0),1)    
    
def get_epsilon_c(i, f, b, alpha):#i,f,b必须为int
    tmp1,tmp2 = i - (alpha * f + (1-alpha) * b),f-b
    tmp3,tmp4 = sum(tmp1 * tmp1), sum(tmp2 * tmp2)
    return tmp3/tmp4


def get_prior(prior,XF_leaf,XB_leaf,testing_leaf,k_F, k_B):
    rows, cols= prior.shape
    num_s, num_tree = testing_leaf.shape    
    FLeaf_count = [[0] for i in range(num_tree)]
    BLeaf_count = [[0] for i in range(num_tree)]
    FLeaf_ind, BLeaf_ind = XF_leaf.T, XB_leaf.T
    for i in range(num_tree):
        FLeaf_count[i] = collections.Counter(FLeaf_ind[i])
        BLeaf_count[i] = collections.Counter(BLeaf_ind[i])
    # per_tree_pred = [tree.predict(X) for tree in clf.estimators_] # 每棵树对每一样本的预测类别
    Leaf_ratio = np.zeros((num_s, num_tree))
    for i in range(num_s):
        for j in range(num_tree):
            node = testing_leaf[i][j]
            num_f, num_b = FLeaf_count[j][node], BLeaf_count[j][node]
            Leaf_ratio[i][j] = num_f * k_F[j] / (num_f * k_F[j] + num_b * k_B[j])

    '''
    per_testing_prob = []
    filt = 1 / 510
    for i in range(num_s):
        item = sum(Leaf_ratio[i]) / num_tree

        if item < filt:
            item = 0
        elif item > 1 - filt:
            item = 1

        per_testing_prob.append(item)
    '''
    count,step=0,0
    for i in range(rows):
        for j in range(cols):
            if prior[i,j] not in [0,1]:
                alpha=sum(Leaf_ratio[step]) / num_tree
                #alpha=per_testing_prob[step]#[1]
                step=step+1
                if alpha>0 and alpha<1:
                    count=count+1
                prior[i][j]=alpha
    print("混合值个数：",count)
    return prior

import math
def get_Gaus(pix1,pix2,id1,id2,sig_r,sig_s):
    r,s=sum((pix1-pix2)**2),(id1[0]-id2[0])**2+(id1[1]-id2[1])**2
    g_r,g_s=math.exp(-0.5*r/sig_r),math.exp(-0.5*s/sig_s)
    return g_r*g_s


def MBF2Matting(Img,alpha,unknown_ind,dist,sig_r,sig_s):
    # alpha按[0,1]
    alpha1=alpha.astype(float)
    filt=1/510
    rows,cols=alpha.shape
    for i in range(rows):
        for j in range(cols):
            if unknown_ind[i,j]==128: # 对未知区域预测的alpha值修正
                pix1,id1=Img[i,j],[i,j]
                wGaus,Gaus=[],[]
                x,y=i-(dist-1)/2,j-(dist-1)/2
                x,y=int(x),int(y)
                for m in range(dist): # 按(i,j)为中心，从左上角点(x,y)开始遍历
                    for n in range(dist):
                        new_i,new_j=x+m,y+n
                        if 0<=new_i<rows and 0<=new_j<cols:
                            pix2,id2=Img[new_i,new_j],[new_i,new_j]
                            Gaus_value=get_Gaus(pix1,pix2,id1,id2,sig_r,sig_s)
                            Gaus.append(Gaus_value)
                            wGaus.append(Gaus_value*alpha1[new_i,new_j])
                if sum(Gaus)==0:
                    print(id1)                 
                alpha1[i,j]=sum(wGaus)/sum(Gaus)
                
    for i in range(rows):
        for j in range(cols):
            if unknown_ind[i,j]:
                if alpha1[i,j]<filt:# 对未知区域预测的alpha值修正
                    alpha1[i,j]=0
                elif alpha1[i,j]>1-filt:
                    alpha1[i,j]=1
    return alpha1    

    
def get_leaf(Img,clf,Trimap):   
    prior=Trimap/255
      
    unknown_seq=[(i,j) for i in range(rows) for j in range(cols) if Trimap_real[i][j]==128]
    min_rows,min_cols=min(item[0] for item in unknown_seq),min(item[1] for item in unknown_seq)
    max_rows,max_cols=max(item[0] for item in unknown_seq),max(item[1] for item in unknown_seq)
    scale=[min_rows,max_rows,min_cols,max_cols]    
    Trimap_mid=np.ones_like(Trimap)
    for i in range(rows):
        for j in range(cols):
            if scale[0]<=i<=scale[1] and scale[2]<=j<=scale[3]:
                Trimap_mid[i,j]=Trimap[i,j]
          
    foreground_ind = Trimap_mid == 255
    background_ind = Trimap_mid == 0
    
    #foreground_ind = Trimap == 255
    #background_ind = Trimap == 0
    unknown_ind = Trimap == 128
            
    Img_feat=get_Imgfeat(Img)
    #X,Y=get_XY(Img_feat,foreground_ind,background_ind)
    X_F,X_B,testing=Img_feat[foreground_ind],Img_feat[background_ind],Img_feat[unknown_ind]
    X=np.vstack((X_F,X_B))    
    foreground_label,background_label=np.ones((len(X_F),1), dtype='float'),np.zeros((len(X_B),1), dtype='float')
    Y=np.vstack((foreground_label,background_label))
    l_F= len(X_F)
    
    # 克服类别不平衡问题，使用上采样方法
    from sklearn.utils import resample
    FY,BY=np.where(Y==1)[0],np.where(Y==0)[0]
    X_upsampled, y_upsampled = resample(X[FY], Y[FY],
                                        replace=True,
                                        n_samples=X[BY].shape[0],
                                        random_state=40)   
    clf = clf.fit(X,Y.reshape((len(Y),)))
    clf = clf.fit(X,Y.reshape((len(Y),)))
    
    #Img_F,Img_B,Img_testing=Img[foreground_ind],Img[background_ind],Img[unknown_ind]
    xy_Img=np.zeros(shape=(rows,cols,2),dtype=int)
    for i in range(rows):
        for j in range(cols):
            xy_Img[i,j]=[i,j]
    #xy_F, xy_B, xy_testing=xy_Img[foreground_ind],xy_Img[background_ind],xy_Img[unknown_ind]
    Imgxy_F, Imgxy_B, Imgxy_testing=np.hstack((Img[foreground_ind],xy_Img[foreground_ind])),np.hstack((Img[background_ind],xy_Img[background_ind])),np.hstack((Img[unknown_ind],xy_Img[unknown_ind]))

    sample=clf.sample_indices(len(Y))
    XF_sample,XB_sample=[],[]
    for i in range(len(sample)):
        item=np.unique(sample[i])
        itemF,itemB=item[np.where(item<len(X_F))],item[np.where(item>=len(X_F))]
        XF_sample.append(itemF)
        XB_sample.append(itemB)
 
    XF_leaf=clf.apply(X_F) # 叶子节点下标
    XB_leaf=clf.apply(X_B) # 叶子节点下标
    testing_leaf=clf.apply(testing)
    num_s, num_tree = testing_leaf.shape
    
    insaplF,insaplB=np.zeros_like(XF_leaf),np.zeros_like(XB_leaf)
    k_F, k_B = np.zeros(shape=(num_tree, 1)), np.zeros(shape=(num_tree, 1))
    for j in range(len(sample)):
        n_F, n_B = 0, 0
        for i in XF_sample[j]:
            insaplF[i][j] = 1
            n_F = n_F + 1
        for i in XB_sample[j]:
            insaplB[i - l_F][j] = 1
            n_B = n_B + 1
        k_F[j], k_B[j] = n_B / (n_F + n_B), n_F / (n_F + n_B)                     
    XF_leaf,XB_leaf=XF_leaf*insaplF,XB_leaf*insaplB
         
    prior=get_prior(prior,XF_leaf,XB_leaf,testing_leaf,k_F, k_B)
    return prior,XF_leaf,XB_leaf,testing_leaf,Imgxy_F, Imgxy_B, Imgxy_testing


def get_FBset(dsam_obj,max_d):
    
    low_bd=thup if max_d>thup else 0
    obj_set=list(np.argsort(dsam_obj)[-1*num_each[0]:])
    st_min=dsam_obj[obj_set[0]]
    gap=st_min-low_bd
    if gap<=0:
        if low_bd==thup:
            return obj_set
        else:
            return np.where(dsam_obj>0)[0]
    else:
        dgs=[1,2/3,1/3,0]
        for t in range(3):
            try:
                temp=random.sample(list(np.where((low_bd+dgs[t+1]*gap<dsam_obj)&(dsam_obj<=low_bd+dgs[t]*gap))[0]),num_each[t+1])
            except:
                temp=list(np.where((low_bd+dgs[t+1]*gap<dsam_obj)&(dsam_obj<=low_bd+dgs[t]*gap))[0])
            obj_set=obj_set+temp
        if len(obj_set)>333:
            print('error')
        obj_set=obj_set[15:]+obj_set[:15]
        return obj_set

def get_FBset_old(dsam_obj,max_d):
    if max_d>=thup:
        return np.argsort(dsam_obj)[-1*n_smp:]
    else:
        candi_obj=np.where(dsam_obj>0)[0]
        if len(candi_obj)<=330:
            return candi_obj
        else:
            obj_set=list(np.argsort(dsam_obj)[-1*num_each[0]:]) #np.argsort(dsam_obj)[-1*num_each[0]:]#[]
            for dg in [2,1,0]:
                try:
                    temp=random.sample(list(np.where((0.25*dg*max_d<dsam_obj)&(dsam_obj<=0.25*(dg+1)*max_d))[0]),num_each[3-dg])
                except:
                    temp=list(np.where((0.25*dg*max_d<dsam_obj)&(dsam_obj<=0.25*(dg+1)*max_d))[0])
                obj_set=obj_set+temp
            obj_set=obj_set[5:]+obj_set[:5]
            return obj_set

import scipy.linalg as linalg
def matting(k):
    #print(k)
    if k%10000==0:
        print("进度："+str(k/n_test * 100)[:5] + '%') 
    i,j=unknown_seq[k]
    obj_leaf=testing_leaf[k].reshape(1,n_tree)
    leaf2F= XF_leaf==obj_leaf
    leaf2B= XB_leaf==obj_leaf
    #nullF,nullB= XF_leaf==zeros,XB_leaf==zeros
    #dsam_F,dsam_B=sum(leaf2F.T)/(n_tree-sum(nullF.T)),sum(leaf2B.T)/(n_tree-sum(nullB.T))
    max_dF,max_dB=max(sum(leaf2F.T))/n_tree,max(sum(leaf2B.T))/n_tree
    if max_dF==1:
        #return [i,j,1,max_dF,max_dB]
        #return 0
    elif max_dB==1:
        return [i,j,0,max_dF,max_dB]
        #return 1
    else:
        Fset,Bset=get_FBset(dsam_F,max_dF),get_FBset(dsam_B,max_dB) 
        pixy=Imgxy_testing[k]
        Fpixy_seq,Bpixy_seq=Imgxy_F[Fset],Imgxy_B[Bset]
        l_F,l_B=len(Fpixy_seq),len(Bpixy_seq)
        e_F,e_B=np.ones(shape=(l_F,1)),np.ones(shape=(1,l_B))
        ### 计算alpha
        F=np.hstack((Fpixy_seq[:,:3],-1*e_F))
        B=np.tile(np.eye(3),l_B)
        B=np.vstack((B,Bpixy_seq[:,:3].reshape(1,-1)))
        subtrc_FB=F@B
        alpha_normsq=np.linalg.norm(subtrc_FB.reshape(-1,3),axis=1).reshape(-1,l_B)
        alpha_normsq[alpha_normsq==0]=MIN_F
        subtrc_PB=pixy[:3]-Bpixy_seq[:,:3]
        alpha_mid=subtrc_PB[0].reshape(-1,1)
        for item in subtrc_PB[1:]:
            alpha_mid=linalg.block_diag(alpha_mid,item.reshape(-1,1))      
        alpha_mid=subtrc_FB@alpha_mid       
        alpha_mat=alpha_mid/(alpha_normsq*alpha_normsq)
        alpha_mat[alpha_mat<0]=0
        alpha_mat[alpha_normsq<=MIN_F]=0
        alpha_mat[alpha_mat>1]=1        
        #b_t=time() 
        alpha_matrav=alpha_mat.ravel()
        cost_mid=np.tile(alpha_matrav,3).reshape(3,-1)
        cost_mid=subtrc_FB.reshape(-1,3)*cost_mid.T
        cost_mat=np.tile(subtrc_PB.astype(float),(l_F,1))
        cost_mat-=cost_mid
        cost_mid=np.linalg.norm(cost_mat,axis=1)
        cost_mat=cost_mid.reshape(l_F,l_B)
        cost_mat/=alpha_normsq
        xy_dF,xy_dB=pixy[-2:]-Fpixy_seq[:,-2:],pixy[-2:]-Bpixy_seq[:,-2:]
        Fdist,Bdist=np.linalg.norm(xy_dF,axis=1),np.linalg.norm(xy_dB,axis=1)
        #Fdist_mean,Bdist_mean=np.mean(Fdist),np.mean(Bdist)
        #nlzFdist,nlzBdist=Fdist/Fdist_mean,Bdist/Bdist_mean
        rl_c,rl_dF,rl_dB=np.mean(cost_mat),np.mean(Fdist),np.mean(Bdist)
        
        cost_mat,Fdist,Bdist=cost_mat/rl_c,Fdist/rl_dF,Bdist/rl_dB
        #print(k)
        
        Fdist_mat,Bdist_mat=Fdist.reshape(-1,1)@e_B,e_F@Bdist.reshape(1,-1)
        cost_mat=cost_mat+(Fdist_mat+Bdist_mat)*0.75
        Mincost,best_alpha=np.min(cost_mat),alpha_matrav[np.argmin(cost_mat)]
        
        best_alpha,Mincost=prior[i][j],MAX_INT32
        for q in range(l_F):
            th=0.15 if 0.2<=best_alpha<=0.8 else 0.1
            item_alpha,item_cost=alpha_mat[q],cost_mat[q]
            select=(-th<=item_alpha-best_alpha)&(item_alpha-best_alpha<=th)
            if np.sum(select)>0:
                cost=np.min(item_cost[select])
                if cost<Mincost:
                    Mincost,best_alpha=cost,item_alpha[np.where(item_cost==cost)][0]
        '''
        Fnorm,Bnorm=np.linalg.norm(Fpixy_seq[:,:3],axis=1),np.linalg.norm(Bpixy_seq[:,:3],axis=1)
        Fcost,Bcost=np.linalg.norm(Fpixy_seq[:,:3]-pixy[:3],axis=1)/Fnorm,np.linalg.norm(subtrc_PB,axis=1)/Bnorm
        Fcost,Bcost=Fcost/rl_c,Bcost/rl_c
        Fcost,Bcost=Fcost+0.75*Fdist*(1+rl_dF/rl_dB),Bcost+0.75*Bdist*(1+rl_dB/rl_dF)
        MinFcost,MinBcost=np.min(Fcost),np.min(Bcost)
        if MinFcost<Mincost and prior[i][j]>0.5: #
            best_alpha=1
        elif MinBcost<Mincost and prior[i][j]<0.5: #
            best_alpha=0
        ''''''
        Img_sm=np.copy(Img)
        Img_sm[pixy[-2],pixy[-1]]=np.array([0,255,255])
        F_seq,B_seq=Imgxy_F[Fset],Imgxy_B[Bset]
        for item in F_seq:
            ind=item[-2:]
            Img_sm[ind[0],ind[1]]=np.array([255,255,0])
        for item in B_seq:
            ind=item[-2:]
            Img_sm[ind[0],ind[1]]=np.array([255,0,0])           
        #show(Img_sm.astype(int))
        cv2.imwrite("ttry.png" , Img_sm)
        bestf,bestb=np.argwhere(cost_mat==np.min(cost_mat))[0]
        Fpixy_seq[bestf][-2:],Bpixy_seq[bestb][-2:]
        '''    
        return best_alpha

def pre_proces(Trimap,thr_C,thr_E,region=None):
    Trimap_mid2=Trimap.astype(float)
    for i in range(rows):
        for j in range(cols):
            if Trimap_mid2[i,j] not in [0,255,128]:
                Trimap_mid2[i,j]=128 
    
    if region==None:
        for i in range(rows):
            for j in range(cols):
                if Trimap_mid2[i,j]==128:
                    for m in range(thr_E):
                        for n in range(thr_E):
                            dist=m**2+n**2
                            dist=np.sqrt(dist)
                            if dist<=thr_E:
                                for (x,y) in [(i+m,j+n),(i+m,j-n),(i-m,j+n),(i-m,j-n)]:
                                    #x,y=i+m,j+n
                                    if 0<=x<rows and 0<=y<cols and Trimap_mid2[x,y]!=128 and sum((Img[x,y]-Img[i,j])**2)<=(thr_C-dist)**2:#if 0<=x<rows and 0<=y<cols and Trimap_mid2[x,y]!=128 and sum((Img[x,y]-Img[i,j])**2)<=(thr_C-dist)**2:
                                        Trimap_mid2[i,j]=Trimap_mid2[x,y] 
    else:
        for i in range(rows):
            for j in range(cols):
                if Trimap_mid2[i,j]==128:
                    for m in range(thr_E):
                        for n in range(thr_E):
                            dist=m**2+n**2
                            dist=np.sqrt(dist)
                            if dist<=thr_E:
                                for (x,y) in [(i+m,j+n),(i+m,j-n),(i-m,j+n),(i-m,j-n)]:
                                    #x,y=i+m,j+n
                                    if 0<=x<rows and 0<=y<cols and Trimap_mid2[x,y]==region and sum((Img[x,y]-Img[i,j])**2)<=(thr_C-dist)**2:#if 0<=x<rows and 0<=y<cols and Trimap_mid2[x,y]!=128 and sum((Img[x,y]-Img[i,j])**2)<=(thr_C-dist)**2:
                                        Trimap_mid2[i,j]=region#Trimap_mid2[x,y] 
    return Trimap_mid2

def priormatting(Img,Trimap_real,Img_file):
    global unknown_seq,testing_leaf,n_test,n_tree,XF_leaf,XB_leaf,zeros,prior,thup,thdn,thdn2,n_smp,Imgxy_testing,Imgxy_F,Imgxy_B,num_each     
    #Trimap_real = cv2.imread(Trimap_dir, cv2.IMREAD_GRAYSCALE)    
    #show(Img)
    Trimap=pre_proces(Trimap_real,thr_C=4,thr_E=3)
    show(Trimap)
    clf = RandomForestClassifier(n_estimators=200,criterion="entropy")#
    prior,XF_leaf,XB_leaf,testing_leaf,Imgxy_F, Imgxy_B, Imgxy_testing=get_leaf(Img,clf,Trimap)
    show(prior)
    #print("prior MSE",compute_mse_loss(prior*255, Alpha, Trimap_real))
    thup,thdn,thdn2=0.5,0.05,0.02
    n_smp,zeros,num_each=300,np.zeros(shape=(1,200),dtype=(int)),[180,75,45,30]#[50,70,80,100]# #60为佳
    n_test,n_tree=testing_leaf.shape
    alpha_samp=Trimap/255
    print("n_test:",n_test,"num_each:",num_each)
    unknown_seq=[(i,j) for i in range(rows) for j in range(cols) if Trimap[i][j]==128]
    pred_seq=Parallel(n_jobs=-1)(map(delayed(matting),list(range(n_test))))
    #pred_seq=[matting(k) for k in range(n_test)]
    for k in range(n_test):
        alpha_samp[unknown_seq[k]]=pred_seq[k]
    
    alpha_weight=(0.5*(prior+alpha_samp)).astype(float)
    alpha_samp=MBF2Matting(Img,alpha_samp,Trimap,dist=3,sig_r=100,sig_s=100)  
    alpha_weight=MBF2Matting(Img,alpha_weight,Trimap,dist=3,sig_r=100,sig_s=100)
    alpha_samp,alpha_weight=alpha_samp*255,alpha_weight*255

    Trimap_real = cv2.imread(Trimap_dir, cv2.IMREAD_GRAYSCALE)    
    print("samp MSE",compute_mse_loss(alpha_samp, Alpha, Trimap_real))
    print("weight MSE",compute_mse_loss(alpha_weight, Alpha, Trimap_real))
    
    return (alpha_samp,alpha_weight)

    print("prior MSE",compute_mse_loss(prior*255, Alpha, Trimap_real))

Trimap_dif=Trimap_real!=Trimap
ttry=prior[Trimap_dif]
np.mean((abs(prior[Trimap_dif]-Alpha[Trimap_dif]/255))**2)

Alpha_seq=Alpha[Trimap==128]
prior_seq=prior[Trimap==128]
error=[]
for i in range(rows):
    for j in range(cols):
        if Trimap[i,j]==128:
            tmp=prior[i,j]
            if tmp<0.0005 or tmp>0.9995:
                error.append(abs(tmp-Alpha[i,j]/255))
np.mean([i**2 for i in error ])

ALpha_gttry=[[i,j,Alpha[item[0],item[1]]/255]for item in ttry]
ttry=Parallel(n_jobs=-1)(map(delayed(matting),list(range(n_test))))
error=[abs(item[-1]-Alpha[item[0],item[1]]/255) for item in ttry]
np.mean([i**2 for i in error])
np.max([i**2 for i in error])

error=[i for i in range(len(ttry)) if abs(ttry[i][-1]-ALpha_gttry[i][-1])>0.05]
tmp=[[ttry[i][-1],ALpha_gttry[i][-1]] for i in error]
np.argmax(np.fromiter(((ttry[i][-1]-ALpha_gttry[i][-1])**2 for i in range(len(ttry))) ,dtype=float)    )
#prior_F,prior_B=np.zeros_like(Trimap_real,dtype=bool),np.zeros_like(Trimap_real,dtype=bool)
#np.sum(sum(prior_B))
'''(ttry[i][-1]-ALpha_gttry[i][-1])**2for i in range(len(ttry))
samp MSE 0.023164887686173263
weight MSE 0.02594973324660128
prior MSE 0.035250284829345946
prior_fb MSE 0.02642444478851888
'''
error=[]
for i in range(rows):
    for j in range(cols):
        if Trimap[i,j]==128:
            tmp=prior[i,j]
            if tmp<0.0005:
                Trimap[i,j]=0
            elif tmp>0.9995:
                Trimap[i,j]=255
def line_coordinates(point1, point2, num_points=100):
    x1, y1 = point1
    x2, y2 = point2
    x_values = np.linspace(x1, x2, num_points)
    y_values = np.linspace(y1, y2, num_points)
    x_values = np.round(x_values).astype(int)
    y_values = np.round(y_values).astype(int)
    coordinates = list(zip(x_values, y_values))
    unique_coordinates = list(dict.fromkeys(coordinates))
    return unique_coordinates

def get_Imgfeat(Img):
    Img_gray=cv2.cvtColor(Img,cv2.COLOR_BGR2GRAY)
    Img_grad_x,Img_grad_y = cv2.Sobel(Img_gray, cv2.CV_32F, 1, 0),cv2.Sobel(Img_gray, cv2.CV_32F, 0, 1)
    Img_gradx, Img_grady = cv2.convertScaleAbs(Img_grad_x), cv2.convertScaleAbs(Img_grad_y)   
    Img_grad = cv2.addWeighted(Img_gradx,0.5, Img_grady, 0.5, 0)

    
    gaussianBlur = cv2.GaussianBlur(Img_gray, (3,3), 0)
    _, binary = cv2.threshold(gaussianBlur, 127, 255, cv2.THRESH_BINARY)
    dst = cv2.Laplacian(binary, cv2.CV_16S, ksize = 3)
    Img_lap = cv2.convertScaleAbs(dst)
    
    Img_lbp_unfrevolve= lbp_uniform(Img_gray)
    Img_gray=Img_gray.astype(int)
   
    Img_lbp=np.array(Img_gray,dtype='uint8')
    Img_lbp=get_lbp(Img_lbp)
    Img_lbp=Img_lbp.reshape(rows,cols,1)
    Img_id=np.zeros((rows,cols,2))

    for i in range(rows):
        for j in range(cols):
            Img_id[i,j][0]=i
            Img_id[i,j][1]=j
    Img_hsv = cv2.cvtColor(Img, cv2.COLOR_BGR2HSV)
    Img_lab = cv2.cvtColor(Img, cv2.COLOR_BGR2LAB)
    
    from skimage import  exposure
    Img_hog = ft.hog(Img_gray,orientations=9,pixels_per_cell=[16,16],cells_per_block=[2,2],visualize=True)[1]
    Img_hog= exposure.rescale_intensity(Img_hog, in_range=(0, 10))
    Img_hog=Img_hog.reshape(rows,cols,1)
    Img_gray=Img_gray.reshape(rows,cols,1)
    Img_grad=Img_grad.reshape(rows,cols,1)
    Img_lap=Img_lap.reshape(rows,cols,1)
    Img_lbp_unfrevolve=Img_lbp_unfrevolve.reshape(rows,cols,1)
    Img_seg=seg_alpha.reshape(rows,cols,1)
    return  np.concatenate((Img_seg,Img_lab,Img_hsv,Img_id,Img_lap,Img_lbp,Img_lbp_unfrevolve,Img_grad,Img_hog),axis=2)#np.concatenate((Img_lab,Img_hsv,Img_lbp,Img_lbp_unfrevolve,Img_lap,Img_grad,Img_hog),axis=2)#np.concatenate((Img_gray,Img_lab,Img_hsv,Img_id,Img_lbp,Img_var,lab_grad,Img_hog),axis=2)



import openslide
from xml.dom.minidom import parse
#### main
if __name__ == "__main__":
    slide_path="D:\\BangweiYe\\Medical_database\\tumor_cell_ann\\"
    slide_files= os.listdir(slide_path)
    slide_file='2175851026_224515.svs'
    xml_file=slide_path+slide_file[:-3]+'xml'     
    slide = openslide.OpenSlide(slide_path+slide_file)
    level = 0#slide.level_count - 1
    width, height = slide.level_dimensions[level]
    #slide_image = np.array(slide.read_region((0, 0), level, (width, height)))[:, :, :3]
    DOMTree = parse(xml_file)
    collection = DOMTree.documentElement
    Annotations = collection.getElementsByTagName("Annotation")
    #color=16776960        
    Ann_dict,RGBcol=dict(),dict()
    for Annotation in Annotations:
        if len(Annotation.getElementsByTagName('Regions')[0].getElementsByTagName('Region'))<50:
            continue
        LineColor=int(Annotation.getAttribute("LineColor"))
        print(LineColor)
        print(len(Annotation.getElementsByTagName('Regions')[0].getElementsByTagName('Region')))
        Ann_dict[LineColor]=Annotation
        #print(LineColor)
        color,color16=[],hex(LineColor)[2:]
        for idx in [0,2,4]:
            if idx>=len(color16):
                color=[0]+color
            else:
                color.append(int(color16[idx:idx+2],base=16))
        RGBcol[LineColor]=color[::-1]
    Nucleus="tumor"
    if Nucleus=="tumor":
        color_out,color_in=65280,16776960
        save_path=slide_path+"AIAnn\\"+slide_file[:-4]+"\\Nucleus_labeling_tumor\\"
    else:
        color_out,color_in=255,16711935
        save_path=slide_path+"AIAnn\\"+slide_file[:-4]+"\\Nucleus_labeling_normal\\"
    
    Regions = Ann_dict[color_out].getElementsByTagName('Regions')[0]
    Regions=Regions.getElementsByTagName('Region')
    Regions_inner=Ann_dict[color_in].getElementsByTagName('Regions')[0].getElementsByTagName('Region')
    Region=Regions[0]
    Region_Id=int(Region.getAttribute("Id"))
    print(Region_Id)
    Vertices=Region.getElementsByTagName('Vertices')[0]
    Vertexs=Vertices.getElementsByTagName('Vertex')
    contor_out=[]
    minx,miny,maxx,maxy=100000000,100000000,0,0
    for idx,Vertex in enumerate(Vertexs):
        y,x=int(Vertex.getAttribute("Y")),int(Vertex.getAttribute("X"))
        #y=max(y,miny)
        contor_out.append([y,x])
        minx,miny=min(x,minx),min(y,miny)
        maxx,maxy=max(x,maxx),max(y,maxy)
    #miny+=20
    patch_w,patch_h=maxx-minx,maxy-miny
    center=[miny+patch_h//2,minx+patch_w//2]
    slide_patch=np.array(slide.read_region((minx-1,miny-1), level, (patch_w+2, patch_h+2)))[:, :, :3]
    #show(slide_patch)
    Trimap_real=np.zeros_like(slide_patch[:,:,0],dtype=np.uint8)    
    
    slide_anned,alpha_out=np.copy(slide_patch),np.zeros_like(slide_patch[:,:,0],dtype=bool)
    tmp=line_coordinates(contor_out[-1], contor_out[0])
    for idx in range(1,len(contor_out)):
        tmp.extend(line_coordinates(contor_out[idx-1], contor_out[idx])[1:] )
    contor_out=[[x-minx,y-miny] for y,x in tmp]
    for x,y in contor_out:
        slide_anned[y,x]=RGBcol[color_out]
        alpha_out[y,x]=True
    fills,_= cv2.findContours(alpha_out.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  
    alpha_out=cv2.drawContours(alpha_out.astype(np.uint8),fills, -1, (255, 255, 255), -1) 

    Trimap_real[alpha_out==255]=128
    contor_ins=[]
    for m in range(len(Regions_inner)):
        Reg=Regions_inner[m]
        Vertexs=Reg.getElementsByTagName('Vertices')[0].getElementsByTagName('Vertex')
        y,x=int(Vertexs[0].getAttribute("Y")),int(Vertexs[0].getAttribute("X"))
        if x<minx or x>maxx or y<miny or y>maxy:
            continue
        else: 
            contor_in=[]
            for idx,Vertex in enumerate(Vertexs):
                y,x=int(Vertex.getAttribute("Y")),int(Vertex.getAttribute("X"))
                contor_in.append([y,x])
            contor_ins.append(contor_in)
    contor_in=contor_ins[0]
    if len(contor_ins)>1:
        y,x=contor_in[0]
        mindist=(y-center[0])**2+(x-center[1])**2
        for j in range(1,len(contor_ins)):
            candi=contor_ins[j]
            y,x=candi[0]
            dist=(y-center[0])**2+(x-center[1])**2
            if dist<mindist:
                contor_in,mindist,used[used_tmp[j]]=candi,dist,True
  
    tmp=line_coordinates(contor_in[0], contor_in[1])
    for idx in range(2,len(contor_in)):
        tmp.extend(line_coordinates(contor_in[idx-1], contor_in[idx])[1:] )
    contor=[[x-minx,y-miny] for y,x in tmp]
    for x,y in contor:
        Trimap_real[y,x]=255
        slide_anned[y,x]=RGBcol[color_in]
    show(slide_anned)
    show(Trimap_real)
    rows, cols= Trimap_real.shape   
    Img=slide_patch
    Trimap=pre_proces(Trimap_real,thr_C=4,thr_E=3)
    #Trimap=Trimap_real
    show(Trimap)
    clf = RandomForestClassifier(n_estimators=200,criterion="entropy")#
    prior,XF_leaf,XB_leaf,testing_leaf,Imgxy_F, Imgxy_B, Imgxy_testing=get_leaf(Img,clf,Trimap)
    show(prior)
    #print("prior MSE",compute_mse_loss(prior*255, Alpha, Trimap_real))
    thup,thdn,thdn2=0.5,0.05,0.02
    n_smp,zeros,num_each=300,np.zeros(shape=(1,200),dtype=(int)),[180,75,45,30]#[50,70,80,100]# #60为佳
    n_test,n_tree=testing_leaf.shape
    alpha_samp=Trimap/255
    print("n_test:",n_test,"num_each:",num_each)
    unknown_seq=[(i,j) for i in range(rows) for j in range(cols) if Trimap[i][j]==128]
    pred_seq=Parallel(n_jobs=-1)(map(delayed(matting),list(range(n_test))))
    #pred_seq=[matting(k) for k in range(n_test)]
    for k in range(n_test):
        alpha_samp[unknown_seq[k]]=pred_seq[k]
    
    alpha_weight=(0.5*(prior+alpha_samp)).astype(float)
    alpha_samp=MBF2Matting(Img,alpha_samp,Trimap,dist=3,sig_r=100,sig_s=100)  
    alpha_weight=MBF2Matting(Img,alpha_weight,Trimap,dist=3,sig_r=100,sig_s=100)
    alpha_samp,alpha_weight=alpha_samp*255,alpha_weight*255

    Trimap_real = cv2.imread(Trimap_dir, cv2.IMREAD_GRAYSCALE)    
    print("samp MSE",compute_mse_loss(alpha_samp, Alpha, Trimap_real))
    print("weight MSE",compute_mse_loss(alpha_weight, Alpha, Trimap_real))

    '''
    save_path="D:\\BangweiYe\\R2Fmatting\\output\\testing2.2\\"
    cv2.imwrite( os.path.join(save_path+"samp\\"+Trimap_cls,Img_file) , alpha_samp)
    cv2.waitKey(0)
    cv2.imwrite( os.path.join(save_path+"weight\\"+Trimap_cls,Img_file) , alpha_weight)
    cv2.waitKey(0)
    '''
    show(alpha_samp)
    show(alpha_weight)
    
    compute_mse_loss(ps_Alpha, alpha_samp,Trimap_real)
    ttry=np.abs(alpha_samp-ps_Alpha)>50
    show(ttry)

    seg=cv2.imread(os.path.join("D:\\BangweiYe\\Medical_database\\tumor_cell_ann\\AIAnn\\2175851026_224515\\Nucleus_labeling_tumor\\labeling\\", '2175851026_224515_ID1.png') )
    #seg_alpha=mask*1
    show(seg) 
    seg_alpha=np.zeros_like(Trimap_real)
    for i in range(rows):
        for j in range(cols):
            if sum((seg[i,j]-RGBcol[color_out])**2)==0:
                seg_alpha[i,j]=1
    fills,_= cv2.findContours(seg_alpha.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  
    seg_alpha=cv2.drawContours(seg_alpha.astype(np.uint8),fills, -1, (255, 255, 255), -1) 
    show(seg_alpha)
     

import sys
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import_path="D:\\BangweiYe\\Medical_database\\code_refer\\segment-anything-main\\"
sys.path.append(import_path)
from segment_anything import sam_model_registry, SamPredictor#,SamAutomaticMaskGenerator
sam_checkpoint,model_type,device = import_path+"notebooks\\"+"sam_vit_h_4b8939.pth","vit_h","cuda"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
predictor = SamPredictor(sam)
def show_mask(mask, ax, random_color=False):
   if random_color:
       color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
   else:
       color = np.array([30/255, 144/255, 255/255, 0.6])
   h, w = mask.shape[-2:]
   mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
   ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
 
predictor.set_image(Img)            
input_point =  np.array([[400,500],[200,500],[200,300],[500,300]])#np.array([contor[0],contor[len(contor)//2],contor[-1]])
input_label = np.ones_like(input_point[:,0])
masks, scores, logits = predictor.predict(
point_coords=input_point,
point_labels=input_label,
multimask_output=True,
)
for i, (mask, score) in enumerate(zip(masks, scores)):
   plt.figure(figsize=(10,10))
   plt.imshow(Img)
   show_mask(mask, plt.gca())
   show_points(input_point, input_label, plt.gca())
   plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
   plt.axis('off')
   plt.show() 
mask=masks[-1]
show(mask)
seg_alpha=mask*1

mask_erode=cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=3)
Trimap_real[mask_erode==1]=255
show(Trimap_real)
show(mask_erode)
cv2.imwrite("ttry.png",alpha_samp)
   