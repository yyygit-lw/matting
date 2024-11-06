function [bestf_ind,bestb_ind,best_alpha] = EvaluateSamples(pareto_F_bw,pareto_B_bw,...
    F_pareto_color,B_pareto_color,F_pareto_dist,B_pareto_dist,U_color)
%EVALUATESAMPLES 此处显示有关此函数的摘要
%   此处显示详细说明
%% calculate alpha values
UF_pareto_num = nnz(pareto_F_bw);
UB_pareto_num = nnz(pareto_B_bw);
[ind_F_pareto,ind_B_pareto] = ind2sub([UF_pareto_num,UB_pareto_num],1:UF_pareto_num*UB_pareto_num);
FB_pareto_alpha = CalAlphaFromFB(F_pareto_color(:,ind_F_pareto),B_pareto_color(:,ind_B_pareto),U_color');
%% color distortion
FB_pareto_alpha_inrange = FB_pareto_alpha;
FB_pareto_alpha_inrange(FB_pareto_alpha_inrange>1) = 1;
FB_pareto_alpha_inrange(FB_pareto_alpha_inrange<0) = 0;
%Alpha*F
af = bsxfun(@times,FB_pareto_alpha_inrange,F_pareto_color(:,ind_F_pareto));
%(1-Alpha)*B
ab = bsxfun(@times,(1-FB_pareto_alpha_inrange),B_pareto_color(:,ind_B_pareto));
%I-(af+(1-a)b)
Iafab = bsxfun(@minus,U_color',af+ab);
color_distortion = exp(-(sqrt(sum(Iafab.^2))));
%% spatial distance cost
F_dist = exp(-F_pareto_dist(ind_F_pareto')/mean(F_pareto_dist));
B_dist = exp(-B_pareto_dist(ind_B_pareto')/mean(B_pareto_dist));
%% fitness
fitness = (color_distortion'.^0.5).*((B_dist.*F_dist).^2);
[~,ind] = max(fitness);
best_alpha = FB_pareto_alpha_inrange(ind);
bestf_ind=ind_F_pareto(ind);
bestb_ind=ind_B_pareto(ind);
end

