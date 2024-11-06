clc,clear all;

Trimap_name='Trimap1\';
Img_path='D:\BangweiYe\matting\input\input_training_lowres\';
Trimap_path=['D:\BangweiYe\matting\input\trimap_training_lowres\',Trimap_name];
gt_path='D:\BangweiYe\matting\gt_training_lowres\';
pred_path=['D:\BangweiYe\matting\PDMS\result\training\',Trimap_name];

img_list = dir(fullfile(Img_path,'*.png'));
img_namels = {img_list.name}'; 

k=19;
name=img_namels{k};
img_name=[Img_path,name];
Trimap_name=[Trimap_path,name];
pred_name=[pred_path,name];
test=3;
[Ih,Iw,~]=size(imread(img_name));
trimap = imread(Trimap_name);
gt_alpha=rgb2gray(imread([gt_path,name]));
gt_alpha=double(gt_alpha)/255;
record_idx = [];
for i=1:Ih
    for j=1:Iw
        if (trimap(i,j)==128)
            alpha1=PDMS_tiaoshi(img_name,Trimap_name,0,i,j);
            alpha2=PDMS_tiaoshi(img_name,Trimap_name,3,i,j);
            if ( or((gt_alpha(i,j))-alpha1<-0.15,gt_alpha(i,j)-alpha1>0.15) & (abs(alpha2-gt_alpha(i,j))>abs(alpha1-gt_alpha(i,j))+0.05) )
                record_idx = [record_idx;[i,j]];
            end
        end
    end
end

i=114;
j=398;
PDMS_tiaoshi(img_name,Trimap_name,0,i,j);
PDMS_tiaoshi(img_name,Trimap_name,3,i,j);

alpha=PDMS_Demo(img_name,Trimap_name);
imwrite(alpha,pred_name);
figure,imshow(alpha);
%{
for k=1:size(img_namels,1)
    name=img_namels{k};
    img_name=[Img_path,name];
    Trimap_name=[Trimap_path,name];
    pred_name=[pred_path,name];
    alpha=PDMS_Demo(img_name,Trimap_name);
    imwrite(alpha,pred_name);
    figure,imshow(alpha);
end
%}


