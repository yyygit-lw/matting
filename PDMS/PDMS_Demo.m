function alpha = PDMS_Demo(img_name,Trimap_name)
% Pixel-level Discrete Multiobjective Sampling Demo
% Yihui Liang, IAIS, SCUT
% 2019.07.20
% This demo implements the PDMS-based (pixel-level discrete
% multiobjective sampling) matting algorithm. Notice that preprocessing and
% postprocessing are not applied in this demo.
% Reference: Huang H, Liang Y*, Yang X, et al. Pixel-level Discrete
% Multiobjective Sampling for Image Matting[J]. IEEE Transactions on Image
% Processing, 27(8):3739-3751

%% Load image and trimap
raw_img = imread(img_name);
trimap = imread(Trimap_name);

%% LBP feat
imgray=rgb2gray(raw_img);
radius=2;
n=16;
lbp_raw=LBP(imgray,radius,n);
lbp=lbp_raw/(2^n-1);

%% Pixel-level Discrete Multiobjective Sampling
%feature
U_ind = find(trimap ==128);
F_ind = find(trimap ==255);
B_ind = find(trimap ==0);
img = reshape(raw_img,numel(trimap),3);
F_color = img(F_ind,:);
B_color = img(B_ind,:);
U_LBP = lbp(U_ind);
F_LBP = lbp(F_ind);
B_LBP = lbp(B_ind);

[y_map,x_map] = ind2sub(size(trimap),1:numel(trimap));
F_corrdinate = [x_map(F_ind)',y_map(F_ind)'];
B_corrdinate = [x_map(B_ind)',y_map(B_ind)'];

[y,x] = ind2sub(size(trimap),1:numel(trimap));
U_alpha = zeros(length(U_ind),1);
parfor n = 1:length(U_ind)
    U_color = single(img(U_ind(n),:));
    %foreground color sampling
    spatial_cost_FU = pdist2([x(F_ind)',y(F_ind)'],[x(U_ind(n)),y(U_ind(n))]);
    color_cost_FU = pdist2(single(img(F_ind,:)),single(img(U_ind(n),:)));
    texture_cost_FU=pdist2(single(F_LBP),single(lbp(U_ind(n))));
    pareto_F_bw = FDMOt([spatial_cost_FU,color_cost_FU,texture_cost_FU]);
    %background color sampling
    spatial_cost_BU = pdist2([x(B_ind)',y(B_ind)'],[x(U_ind(n)),y(U_ind(n))]);
    color_cost_BU = pdist2(single(img(B_ind,:)),single(img(U_ind(n),:)));
    texture_cost_BU=pdist2(single(B_LBP),single(lbp(U_ind(n))));
    pareto_B_bw = FDMOt([spatial_cost_BU,color_cost_BU,texture_cost_BU]);
    %sample feature extration
    F_pareto_color = single(F_color(pareto_F_bw,:)');
    B_pareto_color = single(B_color(pareto_B_bw,:)');
    F_pareto_dist = spatial_cost_FU(pareto_F_bw);
    B_pareto_dist = spatial_cost_BU(pareto_B_bw);
    %evaluation and alpha estimation
    [bestf_ind,bestb_ind,U_alpha(n)] = EvaluateSamples(pareto_F_bw,pareto_B_bw,...
        F_pareto_color,B_pareto_color,F_pareto_dist,B_pareto_dist,U_color);
    strcat(num2str(round((n/length(U_ind))*100)),'% of matting is completed ...')
end

alpha = trimap;
alpha(U_ind) = U_alpha*255;
%imshow(alpha);
clear all;
end
