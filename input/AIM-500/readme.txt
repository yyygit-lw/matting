Here we propose AIM-500 (Automatic Image Matting-500), the first natural image matting test set, which contains 500 high-resolution real-world natural images from three types foregrounds (Salient Opaque, Salient Transparent/Meticulous, Non-salient), seven categories, and the manually labeled alpha mattes. The Dataset is aimed to aid research efforts in the area of image matting and related topics.

The structure of the dataset AIM-500 is as follows:

AIM-500
├── original (the real-world natural images)
├── mask (the high-resolution manually labelled ground truth)
├── trimap (the trimap of the ground truth mask, used in evaluation)
├── usr (the unified semantic representation of the ground truth mask, serving as the auxiliary input for trimap-based methods)
├── aim_category_type.json (a JSON file consisting of the type and category of each image)
├── AIM-500_Release_Agreement.pdf (the dataset release agreement of AIM-500)
├── readme.txt (the readme and instructions)

The dataset is under MIT license.

All research documents and paper that uses the Dataset will need to acknowledge the use of the data by including an appropriate citation to the following:
Li, Jizhizi, Jing Zhang, and Dacheng Tao. "Deep Automatic Natural Image Matting." IJCAI. 2021.

For any further issues,  please refers to the paper (https://arxiv.org/abs/2107.07235), the GitHub repo (https://github.com/JizhiziLi/AIM) or contacts Jizhizi Li at jili8515@uni.sydney.edu.au. Good luck with your research!