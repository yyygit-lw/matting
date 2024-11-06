function F=featureImage(Im)

sF=9;
%       LAB=vl_xyz2lab(vl_rgb2xyz(Im));
% LAB=mean(Im,3);
    Im=double(Im)/255;
    LAB=Im;
[h,w,ch]=size(Im);

F=zeros(h,w,sF);


filter=fspecial('gaussian',2*10+1,0.1);

If=imfilter(Im,filter);

d = [-1 0 1];
% First order derivatives
Iy = imfilter(If,d,'symmetric','same','conv');
Ix = imfilter(If,d','symmetric','same','conv');

% Second order derivatives
Iyy = imfilter(Iy,d,'symmetric','same','conv');
Ixx = imfilter(Ix,d','symmetric','same','conv');

Ix=mean(abs(Ix),3);
Iy=mean(abs(Iy),3);
Ixx=mean(abs(Ixx),3);
Iyy=mean(abs(Iyy),3);

[s2, s1] = meshgrid(1:w,1:h);


Ixx = abs(Ixx)./max(abs(Ixx(:)));
Iyy = abs(Iyy)./max(abs(Iyy(:)));
Ix = abs(Ix)./max(abs(Ix(:)));
Iy = abs(Iy)./max(abs(Iy(:)));
s1 = s1/max(s1(:));
s2 = s2/max(s2(:));


F(:,:,1)=LAB(:,:,1);
F(:,:,2)=LAB(:,:,2);
F(:,:,3)=LAB(:,:,3);
F(:,:,4)=Ix;
F(:,:,5)=Iy;
F(:,:,6)=Ixx;
F(:,:,7)=Iyy;
F(:,:,8)=s2;
F(:,:,9)=s1;


