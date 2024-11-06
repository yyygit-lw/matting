addpath('DS3_v1.0/DS3_v1.0');
addpath('mtimesx_20110223');
imname='GT02.png';
im=imread(imname);
[Ih,Iw,~]=size(im);
Trimap=imread('GT02_Trimap.png');
alpha=klsparsematting(im,Trimap,15,0.025);
imwrite(alpha,'alpha.png');
figure,imshow(alpha);



