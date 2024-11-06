clear all,clc;
name='GT01.png';
Img_dir=['D:\BangweiYe\matting\input\input_testing_lowres\',name];
Trimap_dir=['D:\BangweiYe\matting\input\trimap_training_lowres\Trimap1\',name];
gtalpha_dir=['D:\BangweiYe\matting\input\gt_training_lowres\',name];
alpha_dir=['D:\\BangweiYe\matting\output\training\samp\Trimap1\',name];
prior_dir=['D:\\BangweiYe\matting\output\training\weight\Trimap1\',name];
Img=imread(Img_dir);
Trimap=imread(Trimap_dir);
alpha=imread(alpha_dir);
prior=imread(prior_dir);
gt=imread(gtalpha_dir);

[rows,cols]=size(Trimap);
conf=255-abs(int32(alpha)-int32(prior));%
for i=1:rows
    for j=1:cols
        if or(Trimap(i,j)==0,Trimap(i,j)==255)
            conf(i,j)=0;
        else
            conf(i,j)=conf(i,j)*2;
        end
    end
end

pack=[] ;
pack(:,:,1) = uint8(alpha) ;
pack(:,:,2) = uint8(conf ) ;
pack(:,:,3) = uint8(Trimap) ;
alpha_new = Fun_SmoothingMatting(Img, pack);
alpha_new=alpha_new*255;
sum(sum(alpha_new~=alpha))

compute_mse_loss(alpha_new,gt,Trimap)
imwrite( uint8(alpha_new),['lapost_',name])
Eva(1,k)=compute_mse_loss(alpha_new,gt,trimap); % MSE
Eva(2,k)=compute_sad_loss(pred,gt,trimap); % SAD
Eva(3,k)=compute_gradient_loss(pred,gt,trimap); % GRAD
Eva(4,k)=compute_connectivity_error(pred,gt,trimap,0.1); % CON
