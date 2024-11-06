import cv2
import os
import numpy as np

#from tqdm import tqdm

from compute_sad_loss import compute_sad_loss
from compute_mse_loss import compute_mse_loss
from compute_gradient_loss import compute_gradient_loss
from compute_connectivity_error import compute_connectivity_error
'''
if __name__ == '__main__':
    GT_DIR = './matting_evaluation/gt_alpha'
    TRI_DIR = './matting_evaluation/trimap'
    RE_DIR = './matting_evaluation/pred_alpha'
    DATA_TEST_LIST = './matting_evaluation/name_list.txt'

    fid = open(DATA_TEST_LIST, 'r')
    names = fid.readlines()

    sad = []
    mse = []
    grad = []
    conn = []
    for name in tqdm(names):
        try:
            imname = name.strip()

            pd = cv2.imread(os.path.join(RE_DIR, imname), cv2.IMREAD_GRAYSCALE)

            gt = cv2.imread(os.path.join(GT_DIR, imname), cv2.IMREAD_GRAYSCALE)
            tr = cv2.imread(os.path.join(TRI_DIR, imname), cv2.IMREAD_GRAYSCALE)

            sad.append(compute_sad_loss(pd, gt, tr))
            mse.append(compute_mse_loss(pd, gt, tr))
            grad.append(compute_gradient_loss(pd, gt, tr))
            conn.append(compute_connectivity_error(pd, gt, tr, 0.1))
        except Exception as e:
            pass

    SAD = np.mean(sad)
    MSE = np.mean(mse)
    GRAD = np.mean(grad)
    CONN = np.mean(conn)

    print('SAD: {:.4f}, MSE: {:.4f}, Grad: {:.4f}, Conn: {:.4f} \n'.format(SAD, MSE, GRAD, CONN))
'''
if __name__ == "__main__":
    path="D:\\学习\\研究生\\matting\\RF2Matting"
    target_dir=os.path.join(path, "GT13GTalpha.png")
    trimap_dir=os.path.join(path, "GT13GTtrimap.png")
    pred_path="D:\\学习\\研究生\\matting\\RF2Matting\\pred_fig"
    sad = []
    mse = []
    grad = []
    conn = []

    gt = cv2.imread(target_dir, cv2.IMREAD_GRAYSCALE)
    tr = cv2.imread(trimap_dir, cv2.IMREAD_GRAYSCALE)

    pred_list=[i for _,_,i in os.walk(pred_path)][0]
    for pred_name in pred_list:
        pred = cv2.imread(os.path.join(pred_path, pred_name), cv2.IMREAD_GRAYSCALE)
        sad.append(compute_sad_loss(pred, gt, tr))
        mse.append(compute_mse_loss(pred, gt, tr))
        grad.append(compute_gradient_loss(pred, gt, tr))
        conn.append(compute_connectivity_error(pred, gt, tr, 0.1))

