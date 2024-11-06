
Img_path='D:\Database\alphamatting\lowers\input_training_lowres\';
Trimap_path='D:\Database\alphamatting\lowers\trimap_training_lowres\Trimap2\';
gt_path='D:\Database\alphamatting\lowers\gt_training_lowres\';
pred_path='D:\Database\alphamatting\lowers\training\samp\Trimap2\';

pred_list = dir(fullfile(pred_path,'*.png'));
pred_name = {pred_list.name}'; 

for k=1:size(pred_name,1)
    name=pred_name{k};
    pred_dir=[pred_path,name];
    pred=imread(pred_dir);
    
    gt_dir=[gt_path,name];
    gt=imread(gt_dir);
    gt=rgb2gray(gt);
    
    trimap_dir=[Trimap_path,name];
    trimap=imread(trimap_dir);
    
    Eva(1,k)=compute_mse_loss(pred,gt,trimap); % MSE
    Eva(2,k)=compute_sad_loss(pred,gt,trimap); % SAD
    Eva(3,k)=compute_gradient_loss(pred,gt,trimap); % GRAD
    Eva(4,k)=compute_connectivity_error(pred,gt,trimap,0.1); % CON
end
columns = {'pred', 'MSE', 'SAD', 'GRAD', 'CON'};
data=table(pred_name,Eva(1,:)',Eva(2,:)',Eva(3,:)',Eva(4,:)','VariableNames', columns);
data_name='EvaRFSM.csv';
writetable(data, [pred_path,data_name]);

%{
columns = {'pred', 'MSE', 'SAD', 'GRAD', 'CON'};
data=table(pred_name,Eva(1,:)',Eva(2,:)',Eva(3,:)',Eva(4,:)','VariableNames', columns);
data_name='RF2matte2.csv';
writetable(data, [pred_path,data_name]);


Img_path='D:\学习\研究生\matting\RF2Matting\';
trimap_dir=[Img_path,'GT13GTtrimap.png'];
trimap=imread(trimap_dir);
gt_dir=[Img_path,'GT13GTalpha.png'];
gt=imread(gt_dir);
gt=rgb2gray(gt);
pred_path=Img_path%'D:\学习\研究生\matting\RF2Matting\pred_GT13\';

%pred_list = dir(fullfile(pred_path,'*.png'));
pred_name = {'GT13_alpha_smp.png','GT13_alpha_smpost.png'}'; 

for k=1:size(pred_name,1)
    name=pred_name{k};
    pred_dir=[pred_path,name];
    pred=imread(pred_dir);
    
    Eva(1,k)=compute_mse_loss(pred,gt,trimap); % MSE
    Eva(2,k)=compute_sad_loss(pred,gt,trimap); % SAD
    Eva(3,k)=compute_gradient_loss(pred,gt,trimap); % GRAD
    Eva(4,k)=compute_connectivity_error(pred,gt,trimap,0.1); % CON
end

columns = {'pred', 'MSE', 'SAD', 'GRAD', 'CON'};
data=table(pred_name,Eva(1,:)',Eva(2,:)',Eva(3,:)',Eva(4,:)','VariableNames', columns);
data_name='EvaGT13.csv';
writetable(data, [pred_path,data_name]);
}%