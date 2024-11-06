
function alpha=klsparsematting(im,Trimap,supsize,gamma)

%   Paper : "Image Matting with KL-Divergence Based Sparse Sampling, ICCV 2015"
%   Author: Levent Karacan, Aykut Erdem, Erkut Erdem 
%            (karacan@cs.hacettepe.edu.tr, aykut@cs.hacettepe.edu.tr, erkut@cs.hacettepe.edu.tr)
%   Date  : 26/04/2016
%   Version : 1.0 
%   Copyright 2016, Hacettepe University, Turkey.
%
%   Output:
%   alpha          : alpha matte extracted from input image.
%
%   Parameters:
%   @im        : Input image.
%   @supsize   : Superpixel size
%   @gamma     : corresponds to alpha parameter in DS3 algorithm which specifies the tradeoff between number of samples and representiveness
  
%   Example
%   ==========
%   alpha=klsparsematting(im,Trimap,15,0.025);
%   Default Parameters (supsize = 15, gamma = 0.025 )



if (~exist('supsize','var'))
    supsize=15;
end
if (~exist('ps','var'))
    gamma=0.025;
end


I=im;
[Ih,Iw,~]=size(im);
double_im=im2double(im);
[h,w,ch]=size(im);

%% SLIC Superpixels using vl_feat computer vision library
imlab = vl_xyz2lab(vl_rgb2xyz(im));
dataLAB_L=reshape(double(imlab(:,:,1)),[h*w 1]);
dataLAB_A=reshape(double(imlab(:,:,2)),[h*w 1]);
dataLAB_B=reshape(double(imlab(:,:,3)),[h*w 1]);
superpixels = vl_slic(single(imlab), supsize, 500);
superpixels=superpixels+1;

%% Visualize superpixels
% [gx,gy] = gradient(double(superpixels));
% segs=(gx.^2+gy.^2);
% ind=find(segs~=0);
% seg_pix=zeros(size(segs));
% seg_pix(ind)=255;
% supims=[];
% supims(:,:,1)=double(im(:,:,1))+double(seg_pix);
% supims(:,:,2)=double(im(:,:,2))+double(seg_pix);
% supims(:,:,3)=double(im(:,:,3))+double(seg_pix);
% figure,imshow(uint8(supims))
%%

lbls=unique(superpixels);
numOfSup=size(lbls,1);
glb_ind=reshape(1:h*w,[h,w]);
[x_ind,y_ind]=meshgrid(1:w,1:h);

RmaskB= (Trimap < 50); 
RmaskF= (Trimap >200 );
Unknown = RmaskB * 5 + RmaskF ;   
Unknown(Unknown==0)=3 ;
RmaskU = (Unknown==3);
RmaskFB=RmaskF&RmaskB;

MaskKnown=[];
MaskKnown(RmaskU)=-1;
MaskKnown(RmaskF)=1;
MaskKnown(RmaskB)=1;
MaskKnown=reshape(MaskKnown,[h,w]);

known_superpixels=double(superpixels).*MaskKnown-1;
lblinds=1:size(lbls,1);

B_superpixels=superpixels(RmaskB);
Bsupim=double(superpixels).*RmaskB;
B_lbls=unique(B_superpixels);
numOfBSup=size(B_lbls,1);

F_superpixels=superpixels(RmaskF);
Fsupim=double(superpixels).*RmaskF;
F_lbls=unique(F_superpixels);
numOfFSup=size(F_lbls,1);

U_superpixels=superpixels(RmaskU);
Usupim=double(superpixels).*RmaskU;
U_lbls=unique(U_superpixels);
numOfUSup=size(U_lbls,1);

feature=featureImage(im);
[covFB,meanFB,covInvFB,covDetFB]=featureStatistic(feature,superpixels,[B_lbls;F_lbls]);
[covU,meanU,covInvU,covDetU]=featureStatistic(feature,superpixels,U_lbls);


dissimilarityType = 'Euc';
p = inf; % norm used for L1/Lp optimization in DS3
regularized = true; % true: regularized version, false: constrained version
gamma=gamma*4;% regularizer coefficient
verbose = true; % true/false: show/hide optimization steps


D2 = KLDivergenceMat(covFB,meanFB,covInvFB,covDetFB,covU,meanU,covInvU,covDetU);
D=1-(1./(D2+0.5));
Ds=1./(D2+0.5);

Known_lbls=[B_lbls;F_lbls];
D=sqrt(real(D).^2+imag(D).^2);
A=D<0;
D(A)=0;

if (regularized)
    [rho_min, rho_max] = computeRegularizer(D,p);
    options.verbose = verbose;
    options.rho = gamma * rho_max; % regularization parameter
    options.mu = 1 * 10^-1;
    options.maxIter = 3000;
    options.errThr = 1 * 10^-7;
    Z = ds3solver_regularized(D,p,options);
else
    options.verbose = verbose;
    options.rho = gamma; % regularization parameter
    options.mu = 1 * 10^-1;
    options.maxIter = 3000;
    options.errThr = 1 * 10^-7;
    Z = ds3solver_constrained(D,p,options);
end

% find representatives
[sInd,repness] = findRepresentatives(Z);

BackReps=sInd<=numOfBSup;
BackRepInd=sInd(BackReps);
BackRepness=repness(BackReps);

ForeReps=BackReps==0;
ForeRepInd=sInd(ForeReps);
ForeRepIndDummy=ForeRepInd;
ForeRepness=repness(ForeReps);
ForeRepInd=ForeRepInd-numOfBSup;

[BSample,repSuperpixelBlbls]=RepresentativeSample(im,Bsupim,lbls,B_lbls,BackRepInd,RmaskB);
[FSample,repSuperpixelFlbls]=RepresentativeSample(im,Fsupim,lbls,F_lbls,ForeRepInd,RmaskF);
[USample,repSuperpixelUlbls]=RepresentativeSample(im,Usupim,lbls,U_lbls,1:size(U_lbls,1),RmaskU);



%% Boundary Region for Local Samples
RMDilaterFilter = ones(4*4+1)>0;
Boundary_B =(RmaskB - imerode(RmaskB,RMDilaterFilter ))>0 ;
Boundary_F =(RmaskF - imerode(RmaskF,RMDilaterFilter ))>0 ;

BoundB_superpixels=superpixels(Boundary_B);
BoundBsupim=double(superpixels).*Boundary_B;
BoundB_lbls=unique(BoundB_superpixels);
numOfBBSup=size(BoundB_lbls,1);

BoundF_superpixels=superpixels(Boundary_F);
BoundFsupim=double(superpixels).*Boundary_F;
BoundF_lbls=unique(BoundF_superpixels);

numOfBFSup=size(BoundF_lbls,1);
[covBB,meanBB,covInvBB,covDetBB]=featureStatistic(feature,superpixels,BoundB_lbls);
[covFF,meanFF,covInvFF,covDetFF]=featureStatistic(feature,superpixels,BoundF_lbls);

DB = KLDivergenceMat(covBB,meanBB,covInvBB,covDetBB,covU,meanU,covInvU,covDetU);
DF = KLDivergenceMat(covFF,meanFF,covInvFF,covDetFF,covU,meanU,covInvU,covDetU);
DBs=1./(0.5+DB);
DFs=1./(0.5+DF);

Boundary_B3=repmat(Boundary_B,[1,1,3]);
Boundary_F3=repmat(Boundary_F,[1,1,3]);
LocalBsmps=reshape(I(Boundary_B3),[sum(Boundary_B(:)),3]);
LocalFsmps=reshape(I(Boundary_F3),[sum(Boundary_F(:)),3]);



[BBSample,BrepSuperpixelBlbls]=RepresentativeSample(im,BoundBsupim,lbls,BoundB_lbls,1:size(BoundB_lbls,1),Boundary_B);
[BFSample,BrepSuperpixelFlbls]=RepresentativeSample(im,BoundFsupim,lbls,BoundF_lbls,1:size(BoundF_lbls,1),Boundary_F);

LocalFlocs=cell2mat(BFSample(:,6));
LocalBlocs=cell2mat(BBSample(:,6));

%% LabelExpansion -------------------------------------------------
ExpThr_U=9/256 ;
ExpThr_D=1/256 ;
ExpThrDist = ExpThr_U-ExpThr_D ;
MaxIterExp=9;
for i=1 : MaxIterExp
    ExpDist =i ;
    ExpThr =ExpThr_U - i* ExpThrDist / MaxIterExp ;
    [RMaskFExp ,RMaskBExp] = LabelExpansion (I, RmaskF,RmaskB , ExpDist, ExpThr)  ;
    i
end

RmaskF =  RMaskFExp ;
RmaskB =  RMaskBExp ;


Unknown = zeros (Ih,Iw) ; 
Unknown = RmaskB * 5 + RmaskF ;
Unknown(Unknown==0)=3 ;
RmaskU = (Unknown==3) ;


FSampleMat=cell2mat(FSample(:,4));
BSampleMat=cell2mat(BSample(:,4));

SuperB=repSuperpixelBlbls;
SuperF=repSuperpixelFlbls;

BInx=zeros(size(SuperB,1),1);
FInx=zeros(size(SuperF,1),1);
for bb=1:size(SuperB,1)
    INI=find([B_lbls;F_lbls]==SuperB(bb));
    BInx(bb)=INI(1);
end
for bb=1:size(SuperF,1)
    INI=find([B_lbls;F_lbls]==SuperF(bb));
    FInx(bb)=INI(1);
end
SupB=zeros(size(SuperB,1),1);
SupF=zeros(size(SuperF,1),1);
for bb=1:size(SuperB,1)
    INI=find(lbls==SuperB(bb));
    SupB(bb)=INI(1);
end
for bb=1:size(SuperF,1)
    INI=find(lbls==SuperF(bb));
    SupF(bb)=INI(1);
end

NumGenSmp4F = size(FSampleMat,1) ;
NumGenSmp4B = size(BSampleMat,1) ;

Plane_F =  reshape(repmat(FSampleMat , [NumGenSmp4B,1]) ,[NumGenSmp4F,NumGenSmp4B,3] ) ;
Plane_B =  reshape(repmat(BSampleMat(:)' , [NumGenSmp4F,1]) ,[NumGenSmp4F,NumGenSmp4B,3] ) ;


KL_Dist=KLDivSimilarity(BackRepInd,ForeRepIndDummy,covFB,meanFB,covInvFB,covDetFB);

SetPlane_F = Plane_F;
SetPlane_B = Plane_B ;


ULbl2F = zeros(Ih,Iw)+1 ;
ULbl2B = zeros(Ih,Iw)+1 ;

ULbl2F(RmaskF)=0 ;
ULbl2F(RmaskB)=0 ;
ULbl2B(RmaskF)=0 ;
ULbl2B(RmaskB)=0 ;




NumIter=0 ;
NumUSmp = sum((RmaskU(:)==1));

B = zeros(size(I));
F = zeros(size(I));
Alpha_raw=  zeros(Ih,Iw) ;
Robust =  zeros(Ih,Iw) ;


DFFBB = KLDivergenceMat(covFF,meanFF,covInvFF,covDetFF,covBB,meanBB,covInvBB,covDetBB);
DFBB = KLDivergenceMat(covFB(:,:,numOfBSup+1:end),meanFB(:,:,numOfBSup+1:end),covInvFB(:,:,numOfBSup+1:end),covDetFB(numOfBSup+1:end),covBB,meanBB,covInvBB,covDetBB);
DFFB = KLDivergenceMat(covFF,meanFF,covInvFF,covDetFF,covFB(:,:,1:numOfBSup),meanFB(:,:,1:numOfBSup),covInvFB(:,:,1:numOfBSup),covDetFB(1:numOfBSup));


DFFBB=1./(0.5+DFFBB);
DFBB=1./(0.5+DFBB);
DFFB=1./(0.5+DFFB);


smpsize=10;
for j=1 : Iw
    [j  Iw]
    for i=1 : Ih
        
        if (RmaskU(i,j)==1)
            
            
            NumIter=NumIter+1 ;
            PInd = (j-1)*Ih + i ;
            
            ulbls=superpixels(i,j);
            ulbl=find(lbls==ulbls);
            UInx=find(U_lbls==ulbls);
            
            
            
            
            %% Local samples
            
            locBdist=sqrt(sum(bsxfun(@minus,[j i],LocalBlocs).^2,2));
            locFdist=sqrt(sum(bsxfun(@minus,[j i],LocalFlocs).^2,2));
            [dBl,IBl]=sort(locBdist,1);
            [dFl,IFl]=sort(locFdist,1);
            BLocalsmps=cell2mat(BBSample(IBl(1:smpsize),4));
            FLocalsmps=cell2mat(BFSample(IFl(1:smpsize),4));
            LocalBl=LocalBlocs(IBl(1:smpsize),:);
            LocalFl=LocalFlocs(IFl(1:smpsize),:);
            %%
            
            KL_Dist1=[KL_Dist;DFFB(IFl(1:smpsize),BackRepInd)];
            KL_Dist2=[DFBB(ForeRepInd,IBl(1:smpsize));DFFBB(IFl(1:smpsize),IBl(1:smpsize))];
            KL_DistN=[KL_Dist1 KL_Dist2];
            FSampleMatAll=[FSampleMat;double(FLocalsmps)];
            BSampleMatAll=[BSampleMat;double(BLocalsmps)];
        
            BackKLsim=[ Ds(BackRepInd,UInx);DBs(IBl(1:smpsize),UInx)];
            BackKLsim=BackKLsim./max(BackKLsim(:));
            ForeKLsim=[ Ds(numOfBSup+ForeRepInd,UInx);DFs(IFl(1:smpsize),UInx)];
            ForeKLsim=ForeKLsim./max(ForeKLsim(:));
              
            NumGenSmp4F = size(FSampleMatAll,1) ;
            NumGenSmp4B = size(BSampleMatAll,1) ;
            
            Plane_F =  reshape(repmat(FSampleMatAll , [NumGenSmp4B,1]) ,[NumGenSmp4F,NumGenSmp4B,3] ) ;
            Plane_B =  reshape(repmat(BSampleMatAll(:)' , [NumGenSmp4F,1]) ,[NumGenSmp4F,NumGenSmp4B,3] ) ;
            
            Plane_FKLs =  reshape(repmat(ForeKLsim , [NumGenSmp4B,1]) ,[NumGenSmp4F,NumGenSmp4B,1] ) ;
            Plane_BKLs =  reshape(repmat(BackKLsim' , [NumGenSmp4F,1]) ,[NumGenSmp4F,NumGenSmp4B,1] ) ;
            
            
            
            %%
            PLInd4F=ULbl2F(i,j);
            PLInd4B=ULbl2B(i,j);
            
            RGBValU(1,1)=double(im(i,j,1));
            RGBValU(1,2)=double(im(i,j,2));
            RGBValU(1,3)=double(im(i,j,3));
            
            
            
            TSelF2DInd=cell2mat(FSample(:,6));
            TSelB2DInd=cell2mat(BSample(:,6));
            TSelB2DIndAll=[TSelB2DInd;LocalBl];
            TSelF2DIndAll=[TSelF2DInd;LocalFl];
            
            
            NumGenSmp4F= size(Plane_F,1) ;
            NumGenSmp4B= size(Plane_F,2) ;
            
            PVal = reshape(double(I(i,j,:)),[1,3]) ;
            Plane_P =  reshape(repmat(PVal ,[NumGenSmp4F*NumGenSmp4B , 1]),[NumGenSmp4F,NumGenSmp4B,3]) ;
            
            
            % Alpha 
            Plane_Alpha = sum((Plane_P - Plane_B).* (Plane_F - Plane_B),3)./ (sum((Plane_F - Plane_B).^2,3)) ;
            
            
            % Chromatic distortion
            Plane_AlphaExt = repmat(Plane_Alpha,[1,1,3]) ;
            
            Plane_Pest = Plane_AlphaExt.* Plane_F+ (1-Plane_AlphaExt).* Plane_B ;
            Plane_DistEst =     sqrt(sum((Plane_P-Plane_Pest).^2,3));
            PlaneW2= exp(-Plane_DistEst) ;
                         
            % Spatial Energy 
            TDist2F= sqrt((TSelF2DIndAll(:,1)-i).^2 +(TSelF2DIndAll(:,2)-j).^2 ) ;
            TDist2B= sqrt((TSelB2DIndAll(:,1)-i).^2 +(TSelB2DIndAll(:,2)-j).^2 ) ;        
            TDist2FPlane = reshape(repmat(TDist2F , [NumGenSmp4B,1]) ,[NumGenSmp4F,NumGenSmp4B] ) ;
            TDist2BPlane = reshape(repmat(TDist2B(:)' , [NumGenSmp4F,1]) ,[NumGenSmp4F,NumGenSmp4B] ) ;
            PlaneW4 = exp(-TDist2FPlane./ mean(TDist2FPlane(:))).*exp(-TDist2BPlane./ mean(TDist2BPlane(:))) ;
            
            % Contextual
            PlaneW3=(Plane_FKLs+Plane_BKLs);
            %% Objective Function Computation ==============================
            ValidTolerance=.2 ;
            PlaneAlphaMask = ones(size(Plane_Alpha)) ;
            PlaneAlphaMask(Plane_Alpha>1+ValidTolerance)=0 ;
            PlaneAlphaMask(Plane_Alpha<0-ValidTolerance)=0 ;
            PlaneAlphaMask(isnan(Plane_Alpha))=0 ;
            
            
            
            %% -----------------------------------------------
            
            FU1=1-D(FInx,UInx);
            BU1=1-D(BInx,UInx);
            
            FUF=DFs(IFl(1:smpsize),UInx);
            BUB=DBs(IBl(1:smpsize),UInx);
            
            FU=[FU1;FUF];
            BU=[BU1;BUB];
            
 
            EWC = [0.5 2 0  0.5 ] ;
            PlaneObj=(PlaneW2.^EWC(2)).*(PlaneW4.^EWC(4)).*PlaneW3.*(KL_DistN.^EWC(1));
            PlaneObj=PlaneAlphaMask.*PlaneObj;
            
            
            FBWeights=[FU;BU]/sum([FU;BU]);
            [FV,FIu]=max(FU);
            [BV,BIu]=max(BU);
            
            
            [TMax , TMaxInd] = max(PlaneObj,[],1) ;   
            [TMax2 , TMaxInd2]= max(TMax)    ;
            SelIndOffsetF =TMaxInd(TMaxInd2)  ; 
            SelIndOffsetB = TMaxInd2          ;  

            if sum(sum(PlaneObj>0))>= 3
                SelAlpha =Plane_Alpha(SelIndOffsetF,SelIndOffsetB);
                SelRobust = (PlaneObj(SelIndOffsetF,SelIndOffsetB));
                  
            else
                SelCAlpha = 0.5 ;
                SelTAlpha=0.5 ;
                SelRobust = 0 ;
                SelAlpha=0.5;
            end
            
            
            Alpha_raw(i,j) = SelAlpha ; 
            SelAlpha=[] ;
            Robust(i,j) = SelRobust ;
            SelRobust=[] ;
            
            F(i,j,:) = Plane_F(SelIndOffsetF,SelIndOffsetB,:);
            B(i,j,:) = Plane_B(SelIndOffsetF,SelIndOffsetB,:);
            
                                      
            strcat(num2str(round((NumIter/NumUSmp)*100)),'% of matting is completed ...')
            
        end 
        
    end
    
end

Alpha_raw(RmaskF)=1 ; 
Alpha_raw(RmaskB)=0 ;

%Refinement Step using matting Laplacian.
RConf1=sqrt(Robust) ;
RConf1(RmaskF)=1 ; RConf1(RmaskB)=1 ;
RConf1=(RConf1) ;
pack=[] ;
pack(:,:,1) = uint8(Alpha_raw*255 ) ;
pack(:,:,2) = uint8((RConf1)*255 ) ;
pack(:,:,3) = uint8(Trimap) ;
alpha = Fun_SmoothingMatting(double(I), pack);






