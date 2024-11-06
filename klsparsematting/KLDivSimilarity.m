function KL= KLDivSimilarity(SuperB,SuperF,cov1,mean1,cov1Inv,cov1Det)


sF=size(cov1,1);
supsizeB=size(SuperB,2);
supsizeF=size(SuperF,2);
unitmat=eye(sF);

KL=zeros(supsizeF,supsizeB);


Bcov1=cov1(:,:,SuperB);
Bmean1=mean1(:,:,SuperB);
Bcov1Inv=cov1Inv(:,:,SuperB);
Bcov1Det=cov1Det(SuperB);

Fcov1=cov1(:,:,SuperF);
Fmean1=mean1(:,:,SuperF);
Fcov1Inv=cov1Inv(:,:,SuperF);
Fcov1Det=cov1Det(SuperF);



for i=1:1:supsizeF
    
    cov1=Fcov1(:,:,i);
    m1=Fmean1(:,:,i);
    Det1=Fcov1Det(i);
    
    InvCOV2=Bcov1Inv(:,:,1:supsizeB);
    InvCOV2=InvCOV2(:,:,:);
    invCov1=Fcov1Inv(:,:,i);
    COV2=Bcov1(:,:,1:supsizeB);
    COV2=COV2(:,:,:);
    InvCOV1=repmat(invCov1,1,supsizeB);
    InvCOV1=reshape(InvCOV1,[sF sF supsizeB]);
    TS=mtimesx(InvCOV1,COV2);
    TS=bsxfun(@times,TS,unitmat);
    TS1=sum(sum(TS,1),2);
    TS1=TS1(:);
    COV1=repmat(cov1,1,supsizeB);
    COV1=reshape(COV1,[sF sF supsizeB]);
    T=mtimesx(InvCOV2,COV1);
    T=bsxfun(@times,T,unitmat);
    
    
    T1=sum(sum(T,1),2);
    T1=T1(:);
    
    mean12=Bmean1(:,:,1:supsizeB);
    MEANDIFFMat=bsxfun(@minus,mean12,m1);
    MEANDIFF=reshape(MEANDIFFMat,[1 sF ,supsizeB]);
    MEANDIFFTRANS=reshape(MEANDIFFMat,[sF 1 ,supsizeB]);
    U=sum(abs(MEANDIFFTRANS(:,:)))./sF;
    dist1=mtimesx(MEANDIFF,InvCOV2);
    
    T2=mtimesx(dist1,MEANDIFFTRANS);
    T2=T2(:);
    distS1=mtimesx(MEANDIFF,InvCOV1);
    TS2=mtimesx(distS1,MEANDIFFTRANS);
    TS2=TS2(:);
    T3=ones(supsizeB,1)*sF;
    DET2=Bcov1Det(1:supsizeB);
    T4=(log(Det1./DET2));
    T4=T4(:);
    
    TS4=log(DET2./Det1)/2;
    TS4=TS4(:); 
    DISTMAT=(T1+T2-T3-T4)/2;
    DISTMATS=(TS1+TS2-T3-TS4)/2;
    DISTMAT=(DISTMAT+DISTMATS)*0.5;
    weight=1./(0.5+DISTMAT);
    KL(i,:)=weight;
end





