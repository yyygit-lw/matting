function [Sample,repSuperpixelXlbls]=RepresentativeSample(im,Xsuperpixels,lbls,X_lbls,repind,RmaskF)
%    Itt=im;

 Sample=[];
 double_im=im2double(im);
  [h,w,~]=size(im);
 [x_ind,y_ind]=meshgrid(1:w,1:h);
 glb_ind=reshape(1:h*w,[h,w]);
    repSuperpixelXlbls=zeros(size(repind,2),1);
    for i=1:size(repind,2)
        [i size(repind,2)]
        supmask=Xsuperpixels==X_lbls(repind(i));
%         repSuperpixelBlbls(i)=X_lbls(repind(i));
%         repSuperpixelXlbls(i)=find(lbls==X_lbls(repind(i))); %%Bug
        repSuperpixelXlbls(i)=lbls(lbls==X_lbls(repind(i)));
        %     supmask=superpixels==B_lbls(sInd(i));
        NumofPix = sum(supmask(:)) ;
        supmask3=repmat(supmask,[1,1,3]);
        RGBsetDouble= reshape(double_im(supmask3), [ NumofPix, 3]);
        RGBSet=reshape(im(supmask3), [ NumofPix, 3]);
        UserValues=reshape(RmaskF(supmask), [ NumofPix, 1]);
        x_loc=x_ind(supmask);
        y_loc=y_ind(supmask);
        glb_loc=glb_ind(supmask);        
        
       
%         r=Itt(:,:,1);
%         g=Itt(:,:,2);
%         b=Itt(:,:,3);
%         r(supmask)=0;
%         g(supmask)=210;
%         b(supmask)=255;
%         r(supmask)=255;
%         g(supmask)=0;
%         b(supmask)=0;
%         Itt(:,:,1)=r;
%         Itt(:,:,2)=g;
%         Itt(:,:,3)=b;
      

        
        
        

        Sample{i,1}=RGBSet;
        Sample{i,2}=[x_loc y_loc];
        Sample{i,3}=glb_loc;
        Sample{i,4}=mean(double(RGBSet),1);
        Sample{i,5}=std(double(RGBSet),0,1);
        Sample{i,6}=mean(double([x_loc y_loc]),1);
        Sample{i,7}=mean(UserValues,1);
    end
%       imwrite(Itt,'sample2.png');