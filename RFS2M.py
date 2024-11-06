# -*- coding: utf-8 -*-
"""
Created on Sun Aug 28 23:56:40 2022

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
MAX_INT32=2147483640

def error_distribution(Alpha,Trimap,alpha_pred,save_path):
    rows, cols = Alpha.shape
    error=[Alpha[i,j]-alpha_pred[i,j] for i in range(rows) for j in range(cols) if Trimap[i,j]==128]

    plt.figure() #初始化一张图
    plt.hist(error,bins=51)  #直方图关键操作
    plt.grid(alpha=0.5,linestyle='-.') #网格线，更好看 
    plt.xlabel('error')
    plt.ylabel('Number of error')
    plt.title(Img_file)
    if save_path!=-1:
        error_filename=Img_file[:-4]+"_error分布图.png"
        plt.savefig(os.path.join(save_path+"error_distribution",error_filename),dpi=500,bbox_inches = 'tight')   # 保存图片 注意 在show()之前  不然show会重新创建新的 图片   
    plt.show()
    
def visual_result(Img,Trimap_dir,alpha_samp,Alpha,save_path,Img_file):
    ## 依次排列：Img,Trimap,prior,alpha_samp,alpha_weight,Alpha
    rows, cols = Alpha.shape
    Img2 = Img[:,:,::-1]
    plt.figure(1)
    plt.subplot(1, 4, 1)
    plt.imshow(Img2)
    plt.xticks([]),plt.yticks([])
    
    plt.figure(1)
    plt.subplot(1,4, 2)
    Trimap_real = cv2.imread(Trimap_dir, cv2.IMREAD_GRAYSCALE)
    plt.imshow(Trimap_real, cmap='gray')
    plt.xticks([]),plt.yticks([])

        
    plt.figure(1)
    plt.subplot(1, 4, 3)
    plt.imshow(alpha_samp, cmap='gray')
    plt.xticks([]),plt.yticks([])
        
    plt.figure(1)
    plt.subplot(1, 4, 4)
    plt.imshow(Alpha, cmap='gray')
    plt.xticks([]),plt.yticks([])
    if save_path!=-1:
        save=os.path.join(save_path+"result",Img_file[:-4]+"_.png")
        plt.savefig( save,dpi=500,bbox_inches = 'tight')
        
    error_distribution(Alpha,Trimap_real,alpha_samp,save_path)


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

def get_Imgfeat(Img):
    rows, cols, _ = Img.shape
    Img_gray=cv2.cvtColor(Img,cv2.COLOR_BGR2GRAY)
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
    lab_grad_x,lab_grad_y = cv2.Sobel(Img_lab, cv2.CV_32F, 1, 0),cv2.Sobel(Img_lab, cv2.CV_32F, 0, 1)
    lab_gradx, lab_grady = cv2.convertScaleAbs(lab_grad_x), cv2.convertScaleAbs(lab_grad_y)
    lab_grad = cv2.addWeighted(lab_gradx,0.5, lab_grady, 0.5, 0)

    Img_var=np.zeros((rows,cols))
    for i in range(rows):
        for j in range(cols):
            Nb_color=[Img[m][n] for (m,n) in [(i-1,j-1),(i-1,j),(i-1,j+1),(i,j-1),(i,j+1),(i+1,j-1),(i+1,j),(i+1,j+1)]
                      if 0<=m<rows and 0<=n<cols]
            Img_var[i][j]=np.var(Nb_color)    
    Img_var=Img_var.reshape(rows,cols,1)
    Img_hog = ft.hog(Img_gray,orientations=12,pixels_per_cell=[20,20],cells_per_block=[2,2],visualize=True)[1]
    Img_hog=Img_hog.reshape(rows,cols,1)
    Img_gray=Img_gray.reshape(rows,cols,1)

    return np.concatenate((Img_gray,Img_lab,Img_hsv,Img_id,Img_lbp,Img_var,lab_grad,Img_hog),axis=2)


def get_estimate_alpha(i, f, b):
    int_i,int_f,int_b = i.astype(int),f.astype(int),b.astype(int)
    tmp1 = int_i - int_b
    tmp2 = int_f - int_b
    tmp3 = sum(tmp1 * tmp2)
    tmp4 = sum(tmp2 * tmp2)
    if tmp4 == 0:
        return 1
    else:
        return min(max(tmp3 / tmp4, 0.0),1)
    
def get_epsilon_c(i, f, b, alpha):
    int_i = i.astype(int)
    int_f = f.astype(int)
    int_b = b.astype(int)
    tmp1 = int_i - (alpha * int_f + (1-alpha) * int_b)
    tmp2 = sum(tmp1 * tmp1)
    return tmp2



def get_testing_Fratio(clf, testing, X_F, X_B):
    Leaf_testing = clf.apply(testing)  # 每个测试样本在每棵树下的叶子节点标号
    num_s, num_tree = Leaf_testing.shape
    l_F, l_B = len(X_F), len(X_B)
    # k_F,k_B=l_B/(l_F+l_B),l_F/(l_F+l_B)

    sample = clf.sample_indices(l_F + l_B)
    XF_sample, XB_sample = [], []
    for i in range(len(sample)):
        item = np.unique(sample[i])
        itemF, itemB = item[np.where(item < l_F)], item[np.where(item >= l_F)]
        XF_sample.append(itemF)
        XB_sample.append(itemB)
    XF_leaf, XB_leaf = clf.apply(X_F), clf.apply(X_B)  # 叶子节点下标
    insaplF, insaplB = np.zeros_like(XF_leaf), np.zeros_like(XB_leaf)
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
    FLeaf_ind, BLeaf_ind = XF_leaf * insaplF, XB_leaf * insaplB  # 每个训练样本在每棵树下的叶子节点标号

    FLeaf_ind, BLeaf_ind = FLeaf_ind.T, BLeaf_ind.T
    Leaf_testing = clf.apply(testing)  # 每个测试样本在每棵树下的叶子节点标号

    FLeaf_count = [[0] for i in range(num_tree)]
    BLeaf_count = [[0] for i in range(num_tree)]
    for i in range(num_tree):
        FLeaf_count[i] = collections.Counter(FLeaf_ind[i])
        BLeaf_count[i] = collections.Counter(BLeaf_ind[i])
    # per_tree_pred = [tree.predict(X) for tree in clf.estimators_] # 每棵树对每一样本的预测类别
    Leaf_ratio = np.zeros((num_s, num_tree))
    for i in range(num_s):
        for j in range(num_tree):
            node = Leaf_testing[i][j]
            num_f, num_b = FLeaf_count[j][node], BLeaf_count[j][node]
            Leaf_ratio[i][j] = num_f * k_F[j] / (num_f * k_F[j] + num_b * k_B[j])

    per_testing_Fratio = []
    filt = 1 / 510
    for i in range(num_s):
        item = sum(Leaf_ratio[i]) / num_tree

        if item < filt:
            item = 0
        elif item > 1 - filt:
            item = 1

        per_testing_Fratio.append(item)
    # per_testing_prob=[sum(Leaf_ratio[i,:])/len(Leaf_ratio[i,:]) for i in range(num_s)]
    return per_testing_Fratio

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
    print("非零个数：",count)
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
    rows, cols= Trimap.shape
    prior=Trimap/255            
    foreground_ind = Trimap == 255
    background_ind = Trimap == 0
    unknown_ind = True ^ np.logical_or(foreground_ind, background_ind)
    
    Img_feat=get_Imgfeat(Img)
    #X,Y=get_XY(Img_feat,foreground_ind,background_ind)
    X_F,X_B,testing=Img_feat[foreground_ind],Img_feat[background_ind],Img_feat[unknown_ind]
    X=np.vstack((X_F,X_B))    
    foreground_label,background_label=np.ones((len(X_F),1), dtype='float'),np.zeros((len(X_B),1), dtype='float')
    Y=np.vstack((foreground_label,background_label))
    l_F= len(X_F)
    
    clf = clf.fit(X,Y)
    
    Img_F,Img_B,Img_testing=Img[foreground_ind],Img[background_ind],Img[unknown_ind]
 
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
    return prior,XF_leaf,XB_leaf,testing_leaf,Img_F,Img_B,Img_testing



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

import random
ttry=np.array([1,3,5,7,9])
ttry=list(ttry)
random.sample(ttry, 3)

def sampred(prior,Trimap,Img_F,Img_B,Img_testing,XF_leaf,XB_leaf,testing_leaf):
    
    rows,cols=Trimap.shape
    alpha_samp=Trimap/255
    n_test,n_tree=testing_leaf.shape
    thup,thdn=n_tree*0.5,n_tree*0.05
    k=0
    for i in range(rows):
        for j in range(cols):
            if Trimap[i][j]==128:
                if k%10000==0:
                    print("进度："+str(k/n_test * 100)[:5] + '%') 
                prealpha=prior[i][j]
                obj_leaf=testing_leaf[k].reshape(1,n_tree)
                leaf2F= XF_leaf==obj_leaf
                leaf2B= XB_leaf==obj_leaf
                nsamF,nsamB=sum(leaf2F.T),sum(leaf2B.T)
                dF,dB=max(nsamF),max(nsamB)
                if np.sum(nsamB>0)==0 or (dF>=thup and dB<=thdn):
                     alpha_samp[i][j]=1
                elif np.sum(nsamF>0)==0 or (dB>=thup and dF<=thdn):
                     alpha_samp[i][j]=0
                else:
                    pix=Img_testing[k]
                    if dF>=thup:
                        Fset=np.argsort(nsamF)[-100:]#random.sample(list(np.where(dF>=thup)[0]),100)
                    else:
                        num_each=int(100/dF)
                        Fset=list(np.where(nsamF==dF)[0])
                        for dg in range(1,dF):
                            try:
                                temp=random.sample(list(np.where(nsamF==dg)[0]),num_each)
                            except:
                                temp=list(np.where(nsamF==dg)[0])
                            Fset=Fset+temp
                    if dB>=thup:
                        Bset=np.argsort(nsamB)[-100:]#random.sample(list(np.where(dB>=thup)[0]),100)
                    else:
                        num_each=int(100/dB)
                        Bset=list(np.where(nsamB==dB)[0])
                        for dg in range(1,dB):
                            try:
                                temp=random.sample(list(np.where(nsamB==dg)[0]),num_each)
                            except:
                                temp=list(np.where(nsamB==dg)[0])
                            Bset=Bset+temp
                    Fpix_ary,Bpix_ary=Img_F[Fset],Img_B[Bset]    
                    Mincost,best_alpha=MAX_INT32,prealpha
                    for fpix in Fpix_ary:
                        for bpix in Bpix_ary:
                            alpha=get_estimate_alpha(pix, fpix, bpix)
                            #if -0.15<alpha-prealpha<0.15:
                            cost=get_epsilon_c(pix, fpix, bpix, alpha)
                            if cost<Mincost:
                                Mincost=cost
                                best_alpha=alpha
                    alpha_samp[i][j]=best_alpha
                k=k+1
     
    alpha_samp=MBF2Matting(Img,alpha_samp,Trimap,dist=3,sig_r=100,sig_s=100)  
    alpha_weight=MBF2Matting(Img,0.5*(prior+alpha_samp),Trimap,dist=3,sig_r=100,sig_s=100)
    alpha_samp,alpha_weight=alpha_samp*255,alpha_weight*255
    return (alpha_samp,alpha_weight)

def pre_proces(Trimap,thr_C,thr_E):   
    rows,cols=Trimap.shape
    for i in range(rows):
        for j in range(cols):
            if Trimap[i,j] not in [0,255,128]:
                Trimap[i,j]=128 
                
    for i in range(rows):
        for j in range(cols):
            if Trimap[i,j]==128:
                for m in range(thr_E):
                    for n in range(thr_E):
                        dist=m**2+n**2
                        dist=np.sqrt(dist)
                        if dist<=thr_E:
                            for (x,y) in [(i+m,j+n),(i+m,j-n),(i-m,j+n),(i-m,j-n)]:
                                #x,y=i+m,j+n
                                if 0<=x<rows and 0<=y<cols and Trimap[x,y]!=128 and sum((Img[x,y]-Img[i,j])**2)<=(thr_C-dist)**2:
                                    Trimap[i,j]=Trimap[x,y] 



def priormatting(Img,Trimap_dir,Img_file):   
    show(Img)     
    #prior= prior/255    
    Trimap = cv2.imread(Trimap_dir, cv2.IMREAD_GRAYSCALE)
    pre_proces(Trimap,thr_C=4,thr_E=3)
    show(Trimap)
    clf = RandomForestClassifier(n_estimators=100,criterion="entropy")#
    prior,XF_leaf,XB_leaf,testing_leaf,Img_F,Img_B,Img_testing=get_leaf(Img,clf,Trimap)

    alpha_samp,alpha_weight=sampred(prior,Trimap,Img_F,Img_B,Img_testing,XF_leaf,XB_leaf,testing_leaf)
    
    save_path="D:\\BangweiYe\\R2Fmatting\\output\\testing2\\"
    cv2.imwrite( os.path.join(save_path+"alpha_samp",Img_file) , alpha_samp)
    cv2.waitKey(0)
    show(alpha_samp)
    cv2.imwrite( os.path.join(save_path+"alpha_weight",Img_file) , alpha_weight)
    cv2.waitKey(0)
    show(alpha_weight)
    
    #Trimap = cv2.imread(Trimap_dir, cv2.IMREAD_GRAYSCALE)
    #MSE1,MSE2=compute_mse_loss(alpha_samp, Alpha, Trimap),compute_mse_loss(alpha_weight, Alpha, Trimap)
    #print("MSE_samp:",MSE1,"MSE_weight:",MSE2)
    #visual_result(Img,Trimap_dir,alpha_samp,Alpha,save_path,Img_file)


#### main
if __name__ == "__main__":
    ## 没有加后处理
    path="D:\\BangweiYe\\matting\\input\\"
    Img_path=path+"input_lowres"
    Img_files= os.listdir(Img_path)
    Trimap_path=path+"trimap_lowres\\Trimap1"
    #Alpha_path=path+"gt_training_lowres"
    #prior_path=path+"alpha_sampost"

    for Img_file in Img_files[5:6]:#到4 即可
        Img_dir=os.path.join(Img_path, Img_file) 
        Img = cv2.imread(Img_dir)
        #Alpha_dir=os.path.join(Alpha_path, Img_file)
        #Alpha=cv2.imread(Alpha_dir, cv2.IMREAD_GRAYSCALE)
        Trimap_dir=os.path.join(Trimap_path, Img_file)
        #prior_dir=os.path.join(prior_path, Img_file)
        #prior=cv2.imread(prior_dir, cv2.IMREAD_GRAYSCALE)
        
        print("matt:"+Img_file)
        priormatting(Img,Trimap_dir,Img_file)#Img, prior,Trimap_dir,Alpha,Img_file






