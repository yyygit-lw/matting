clear all,clc;

%%%% —µ¡∑ºØ
Img_path='D:\Database\alphamatting\lowers\input_training_lowres\';
Trimap_name='Trimap1\';
Trimap_path=['D:\Database\alphamatting\lowers\trimap_training_lowres\',Trimap_name];
alpha_path=['D:\Database\alphamatting\lowers\training\training2.0\samp\',Trimap_name];
prior_path=['D:\Database\alphamatting\lowers\training\training2.0\weight\',Trimap_name];
alpha_list = dir(fullfile(alpha_path,'*.png'));
alpha_name = {alpha_list.name}';
save_path=['D:\Database\alphamatting\lowers\training\de_boosting\lapost\',Trimap_name];
%for k=1:size(alpha_name,1)
for k=1:size(alpha_name,1)
    name=alpha_name{k};
    Img_dir=[Img_path,name];
    Trimap_dir=[Trimap_path,name];
    alpha_dir=[alpha_path,name];
    prior_dir=[prior_path,name];
    Img=imread(Img_dir);
    Trimap=imread(Trimap_dir);
    alpha=imread(alpha_dir);
    prior=imread(prior_dir);
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
    alpha_lapost = Fun_SmoothingMatting(Img, pack);
    alpha_lapost=alpha_lapost*255;
    sum(sum(alpha_lapost~=alpha))
    imwrite( uint8(alpha_lapost),[save_path,name])
end


%%%% ≤‚ ‘ºØ
%{
Trimap_name='Trimap1\';
Img_path='D:\Database\alphamatting\lowers\input_lowres\';
path='D:\Database\alphamatting\lowers\testing\testing2.2\'
Trimap_path=['D:\Database\alphamatting\lowers\trimap_lowres\',Trimap_name];
alpha_path=[path,'samp\',Trimap_name];
prior_path=[path,'samp\',Trimap_name];
alpha_list = dir(fullfile(alpha_path,'*.png'));
alpha_name = {alpha_list.name}';
save_path=[path,'lapost\',Trimap_name];
for k=1:size(alpha_name,1)
    name=alpha_name{k};
    Img_dir=[Img_path,name];
    Trimap_dir=[Trimap_path,name];
    alpha_dir=[alpha_path,name];
    prior_dir=[prior_path,name];
    Img=imread(Img_dir);
    Trimap=imread(Trimap_dir);
    alpha=imread(alpha_dir);
    prior=imread(prior_dir);
    [rows,cols]=size(Trimap);
    conf=255-2*abs(int32(alpha)-int32(prior));%
    for i=1:rows
        for j=1:cols
            if or(Trimap(i,j)==0,Trimap(i,j)==255)
                conf(i,j)=0;
            %else
                %conf(i,j)=conf(i,j);
            end
        end
    end
    pack=[] ;
    pack(:,:,1) = uint8(alpha) ;
    pack(:,:,2) = uint8(conf ) ;
    pack(:,:,3) = uint8(Trimap) ;
    alpha_lapost = Fun_SmoothingMatting(Img, pack);
    alpha_lapost=alpha_lapost*255;
    sum(sum(alpha_lapost~=alpha))
    imwrite( uint8(alpha_lapost),[save_path,name])
end
%}
