# -*- coding: utf-8 -*-
"""
Created on Sat Dec 17 15:54:10 2022

@author: BangweiYe
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
MAX_INT32=2147483640
MIN_F=1E-7

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
    return  np.concatenate((Img_lab,Img_hsv,Img_lap,Img_lbp,Img_lbp_unfrevolve,Img_grad,Img_hog),axis=2)#np.concatenate((Img_lab,Img_hsv,Img_lbp,Img_lbp_unfrevolve,Img_lap,Img_grad,Img_hog),axis=2)#np.concatenate((Img_gray,Img_lab,Img_hsv,Img_id,Img_lbp,Img_var,lab_grad,Img_hog),axis=2)

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
    '''
    foreground_ind = Trimap == 255
    background_ind = Trimap == 0    
    '''
    unknown_ind = Trimap == 128
            
    Img_feat=get_Imgfeat(Img)
    #X,Y=get_XY(Img_feat,foreground_ind,background_ind)
    X_F,X_B,testing=Img_feat[foreground_ind],Img_feat[background_ind],Img_feat[unknown_ind]
    X=np.vstack((X_F,X_B))    
    foreground_label,background_label=np.ones((len(X_F),1), dtype='float'),np.zeros((len(X_B),1), dtype='float')
    Y=np.vstack((foreground_label,background_label))
    l_F= len(X_F)
    
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
    trXF_leaf,trXB_leaf=XF_leaf*insaplF,XB_leaf*insaplB
        
    prior=get_prior(prior,trXF_leaf,trXB_leaf,testing_leaf,k_F, k_B)
    return prior,XF_leaf,XB_leaf,testing_leaf,Imgxy_F, Imgxy_B, Imgxy_testing

def get_alpha(prioralpha,pix,region,region_pix,Img_region,Candi_set,threshold):
    if len(region_pix)==2:
        return (prioralpha,np.array([0,0]))
    Mincost=MAX_INT32
    best_pix=np.array([0,0])
    best_alpha=prioralpha
        
    if region==1:
        for sampb in Candi_set:
            b=Img_region[sampb]
            alpha=get_estimate_alpha(pix, region_pix, b)
            if -1*threshold<alpha-prioralpha<threshold:
                cost=get_epsilon_c(pix, region_pix, b, alpha)
                if cost<Mincost:
                    Mincost=cost
                    best_alpha=alpha
                    best_pix=b
    else:
        for sampf in Candi_set:
            f=Img_region[sampf]
            alpha=get_estimate_alpha(pix, f,region_pix)
            if -1*threshold<alpha-prioralpha<threshold:
                cost=get_epsilon_c(pix, f,region_pix, alpha)
                if cost<Mincost:
                    Mincost=cost
                    best_alpha=alpha
                    best_pix=f
    return (best_alpha,best_pix)

def get_FBset(dsam_obj,max_d):
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
            obj_set=obj_set[3:]+obj_set[:3]
            return obj_set

import scipy.linalg as linalg
def matting(k):
    if k%10000==0:
        print("进度："+str(k/n_test * 100)[:5] + '%') 
    i,j=unknown_seq[k]
    obj_leaf=testing_leaf[k].reshape(1,n_tree)
    leaf2F= XF_leaf==obj_leaf
    leaf2B= XB_leaf==obj_leaf
    nullF,nullB= XF_leaf==zeros,XB_leaf==zeros
    dsam_F,dsam_B=sum(leaf2F.T)/(n_tree-sum(nullF.T)),sum(leaf2B.T)/(n_tree-sum(nullB.T))
    max_dF,max_dB=max(dsam_F),max(dsam_B)
    if max_dF<0.01:
        return 0
    elif max_dB<0.01:
        return 1
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
        '''
        best_alpha,Mincost=prior[i][j],MAX_INT32
        for q in range(l_F):
            th=0.15 if 0.2<=best_alpha<=0.8 else 0.1
            item_alpha,item_cost=alpha_mat[q],cost_mat[q]
            select=(-th<=item_alpha-best_alpha)&(item_alpha-best_alpha<=th)
            if np.sum(select)>0:
                cost=np.min(item_cost[select])
                if cost<Mincost:
                    Mincost,best_alpha=cost,item_alpha[np.where(item_cost==cost)][0]
        ''''''
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
def pre_proces255(Trimap,thr_C,thr_E):
    Trimap_mid2=Trimap.astype(float)
    for i in range(rows):
        for j in range(cols):
            if Trimap_mid2[i,j] not in [0,255,128]:
                Trimap_mid2[i,j]=128 
                
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
                                if 0<=x<rows and 0<=y<cols and Trimap_mid2[x,y]==255 and sum((Img[x,y]-Img[i,j])**2)<=(thr_C-dist)**2:
                                    Trimap_mid2[i,j]=255 
    return Trimap_mid2
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
        count=0
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
                                        count=count+1
        print(count)
    return Trimap_mid2
def pre_matting(Trimap):
    ## 预处理需要参数Trimap=np.copy(Trimap_real)
    Trimap_mid=pre_proces(Trimap_real,thr_C=5,thr_E=4)
    unknown_rows,unknown_cols=np.unique(np.where(Trimap_real==128)[0]),np.unique(np.where(Trimap_real==128)[1])
    foregd_region=np.zeros_like(Trimap_mid)
    foregd_region[unknown_rows[0]:unknown_rows[-1]+1,unknown_cols[0]:unknown_cols[-1]+1]=1
    unknown_ind = Trimap_mid == 128
    foreground_ind = Trimap_mid == 255
    background_ind = (Trimap_mid == 0)*(foregd_region==1)
    Img_feat=get_Imgfeat(Img)  
    ## 首次计算prior扩大已知区域
    clf_pre = RandomForestClassifier(n_estimators=200,criterion="entropy")
    X_F,X_B,testing=Img_feat[foreground_ind],Img_feat[background_ind],Img_feat[unknown_ind]
    X=np.vstack((X_F,X_B))    
    foreground_label,background_label=np.ones((len(X_F),1), dtype='float'),np.zeros((len(X_B),1), dtype='float')
    Y=np.vstack((foreground_label,background_label))
    clf_pre=clf_pre.fit(X,Y.reshape((len(Y),)))
    XF_leaf,XB_leaf,testing_leaf=clf_pre.apply(X_F),clf_pre.apply(X_B),clf_pre.apply(testing) # 叶子节点下标    

    sample=np.array(clf_pre.sample_indices(len(Y)))   
    k_F,n_smp=np.sum(sample<len(X_F),axis=1),len(sample[0])
    k_B=k_F/n_smp
    k_F=1-k_B
    #k_F, k_B = np.ones(shape=(200, 1)), np.ones(shape=(200, 1))

    prior=Trimap_mid/255
    prior=get_prior(prior,XF_leaf,XB_leaf,testing_leaf,k_F, k_B)
    Trimap2=np.copy(Trimap_mid)
    Trimap2[prior>0.99]=255
    Trimap2[prior<0.01]=0
    Trimap2=pre_proces(Trimap2,thr_C=4,thr_E=3)
    prior=Trimap2/255
    ## 获取最终prior以及smp所需输入
    clf = RandomForestClassifier(n_estimators=200,criterion="entropy")#
    unknown_ind = Trimap2 == 128
    foreground_ind = (Trimap2*foregd_region) == 255
    background_ind = (Trimap2*foregd_region) == 0
    X_F,X_B,testing=Img_feat[foreground_ind],Img_feat[background_ind],Img_feat[unknown_ind]
    X=np.vstack((X_F,X_B))    
    foreground_label,background_label=np.ones((len(X_F),1), dtype='float'),np.zeros((len(X_B),1), dtype='float')
    Y=np.vstack((foreground_label,background_label))
    clf.fit(X,Y.reshape((len(Y),)))
    
    xy_Img=np.argwhere(Trimap2!=-1)
    xy_Img=xy_Img.reshape(rows,cols,2)
    Imgxy_F, Imgxy_B, Imgxy_testing=np.hstack((Img[foreground_ind],xy_Img[foreground_ind])),np.hstack((Img[background_ind],xy_Img[background_ind])),np.hstack((Img[unknown_ind],xy_Img[unknown_ind]))
    XF_leaf,XB_leaf,testing_leaf=clf_pre.apply(X_F),clf_pre.apply(X_B),clf_pre.apply(testing) # 叶子节点下标    
    sample=np.array(clf.sample_indices(len(Y)))   
    k_F,n_smp=np.sum(sample<len(X_F),axis=1),len(sample[0])
    k_B=k_F/n_smp
    k_F=1-k_B
    prior=get_prior(prior,XF_leaf,XB_leaf,testing_leaf,k_F, k_B)
    return Trimap2,prior,XF_leaf,XB_leaf,testing_leaf,Imgxy_F, Imgxy_B, Imgxy_testing


def priormatting(Img,Trimap_real,Img_file):
    global unknown_seq,testing_leaf,n_test,n_tree,XF_leaf,XB_leaf,zeros,prior,thup,thdn,thdn2,n_smp,Imgxy_testing,Imgxy_F,Imgxy_B,num_each     
    #Trimap_real = cv2.imread(Trimap_dir, cv2.IMREAD_GRAYSCALE)    
    #Alpha=cv2.imread(os.path.join(path+"gt_training_lowres", Img_file), cv2.IMREAD_GRAYSCALE)    
    #show(Trimap)
    #Trimap=pre_proces(Trimap_real,thr_C=3,thr_E=3,region=0)
    Trimap=pre_proces(Trimap_real,thr_C=4,thr_E=4,region=255)
    '''
    print(np.sum(ps_Alpha[Trimap==0]>1),np.sum(ps_Alpha[Trimap==0]>30))
    print(np.sum(ps_Alpha[Trimap==255]<254),np.sum(ps_Alpha[Trimap==255]<224))
    show((Trimap!=128) &((-5>Trimap-ps_Alpha) |(Trimap-ps_Alpha>5)))
    '''
    clf = RandomForestClassifier(n_estimators=200,criterion="entropy")#
    prior,XF_leaf,XB_leaf,testing_leaf,Imgxy_F, Imgxy_B, Imgxy_testing=get_leaf(Img,clf,Trimap)
    show(prior)
    #print("prior MSE",compute_mse_loss(prior*255, Alpha, Trimap_real))
    thup,thdn,thdn2=0.5,0.05,0.02
    n_smp,zeros,num_each=100,np.zeros(shape=(1,200),dtype=(int)),[65,20,10,5]#[50,70,80,100]# #60为佳
    n_test,n_tree=testing_leaf.shape
    alpha_samp=Trimap/255
    print("n_test:",n_test,"num_each:",num_each)
    unknown_seq=[(i,j) for i in range(rows) for j in range(cols) if Trimap[i][j]==128]
    pred_seq=Parallel(n_jobs=-1)(delayed(matting)(k) for k in range(n_test))
    #pred_seq=[matting(k) for k in range(n_test)]
    for k in range(n_test):
        alpha_samp[unknown_seq[k]]=pred_seq[k]
    
    alpha_weight=(0.5*(prior+alpha_samp)).astype(float)
    alpha_samp=MBF2Matting(Img,alpha_samp,Trimap,dist=3,sig_r=100,sig_s=100)  
    alpha_weight=MBF2Matting(Img,alpha_weight,Trimap,dist=3,sig_r=100,sig_s=100)
    alpha_samp,alpha_weight=alpha_samp*255,alpha_weight*255
    
    #print("samp MSE",compute_mse_loss(alpha_samp, Alpha, Trimap_real))
    #print("weight MSE",compute_mse_loss(alpha_weight, Alpha, Trimap_real))
    
    return (alpha_samp,alpha_weight)

#### main
if __name__ == "__main__":
    #troll√
    path="D:\\BangweiYe\\matting\\input\\"
    Img_path=path+"input_lowres"
    Img_files= os.listdir(Img_path)
    Trimap_cls="Trimap1"
    print(Trimap_cls)
    Trimap_path=path+"trimap_lowres/"+Trimap_cls
    #prior_path=path+"alpha_sampost"
    
    Img_file='troll.png'       
    Img_dir=os.path.join(Img_path, Img_file) 
    Img = cv2.imread(Img_dir)
    show(Img)
    Trimap_dir=os.path.join(Trimap_path, Img_file)
    Trimap_real = cv2.imread(Trimap_dir, cv2.IMREAD_GRAYSCALE)
    rows, cols= Trimap_real.shape    
        #prior_dir=os.path.join(prior_path, Img_file)
        #prior=cv2.imread(prior_dir, cv2.IMREAD_GRAYSCALE)
    ps_Alpha_dir=os.path.join(path+"alphamatting\\"+Trimap_cls, Img_file)
    ps_Alpha = cv2.imread(ps_Alpha_dir, cv2.IMREAD_GRAYSCALE)   
    ps_Alpha[Trimap_real!=128]=Trimap_real[Trimap_real!=128]        
    print("matt:"+Img_file)
    alpha_samp,alpha_weight=priormatting(Img,Trimap_real,Img_file)#Img, prior,Trimap_dir,Alpha,Img_file
    save_path="D:\\BangweiYe\\matting\\output\\testing2.2\\"
    cv2.imwrite( os.path.join(save_path+"samp\\"+Trimap_cls,"a"+Img_file) , alpha_samp)
    cv2.waitKey(0)
    show(alpha_samp)
    cv2.imwrite( os.path.join(save_path+"weight\\"+Trimap_cls,"a"+Img_file) , alpha_weight)
    cv2.waitKey(0)
    show(alpha_weight)


