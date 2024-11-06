function [regcov,regmean,regcovInv,regcovDet]=featureStatistic(F,superpixels,superC)


% F:  Feature Image
% regcov: Region covariances 
[h,w,sF]=size(F);
im1Dind=reshape(1:h*w,[h,w]);
[im2DindW,im2DindH]=meshgrid(1:w,1:h);
lbls=unique(superpixels);
supsize=size(lbls,1);
supsizeC=size(superC,1);
regcov=zeros(sF,sF,supsizeC);
regcovInv=zeros(sF,sF,supsizeC);
regmean=zeros(1,sF,supsizeC);
regcovDet=zeros(supsizeC,1);

for i=1:supsizeC
    
    [i supsizeC]
     ind=find(superpixels==superC(i));
     [indX,indY]=find(superpixels==superC(i));
     
     L=F(:,:,1);
     L=L(ind);    
     a=F(:,:,2);
     a=a(ind);     
     b=F(:,:,3);
     b=b(ind);     
     Ix=F(:,:,4);
     Ix=Ix(ind);     
     Iy=F(:,:,5);
     Iy=Iy(ind);   
     Ixx=F(:,:,6);
     Ixx=Ixx(ind);
     
     Iyy=F(:,:,7);
     Iyy=Iyy(ind);
     
     s1=F(:,:,8);
     s1=s1(ind);
     
     s2=F(:,:,9);
     s2=s2(ind);
     
     sp=im1Dind(ind);
     s1p=im2DindH(ind);
     s2p=im2DindW(ind);
     
     pxsize=size(L,1);
     mat=zeros(sF,pxsize);
        mat(1,:)=L; 
        mat(2,:)=a; 
        mat(3,:)=b; 
        mat(4,:)=Ix;
        mat(5,:)=Iy;
        mat(6,:)=Ixx;
        mat(7,:)=Iyy;
%         mat(8,:)=s1/h;
%         mat(9,:)=s2/w;
         mat(8,:)=s1p/max(s1p(:));
         mat(9,:)=s2p/max(s2p(:));
        
%         mat=mat./sum(mat(:));
%         for kl=1:9
%         mat(kl,:)=mat(kl,:)/sum(mat(kl,:));
%         end
        mmean=mean(double(mat'));
        covM=mat*mat'/(supsize)-mmean'*mmean+0.00000001/(supsize)*eye(sF);
        
        regcov(:,:,i)=covM;
        regmean(:,:,i)=mmean;        
        regcovInv(:,:,i)=(covM)\eye(sF,sF);
        regcovDet(i)=det(covM);
end
