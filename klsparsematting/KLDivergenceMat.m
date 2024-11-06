function [AI] = KLDivergenceMat(covFB,meanFB,covInvFB,covDetFB,covU,meanU,covInvU,covDetU)

sF=size(covFB,1);

supsizeFB=size(covFB,3);
supsizeU=size(covU,3);
unitmat=eye(sF);

AI=zeros(supsizeFB,supsizeU);



for i=1:1:supsizeFB
    
    cov1=covFB(:,:,i);
    m1=meanFB(:,:,i);
    Det1=covDetFB(i);
    
   
    InvCOV2=covInvU(:,:,1:supsizeU);
    InvCOV2=InvCOV2(:,:,:);
    
    
    invCov1=covInvFB(:,:,i);
    COV2=covU(:,:,1:supsizeU);
    COV2=COV2(:,:,:);
    InvCOV1=repmat(invCov1,1,supsizeU);
    InvCOV1=reshape(InvCOV1,[sF sF supsizeU]);
    TS=mtimesx(InvCOV1,COV2);
    TS=bsxfun(@times,TS,unitmat);
    TS1=sum(sum(TS,1),2);
    TS1=TS1(:);

    COV1=repmat(cov1,1,supsizeU);
    COV1=reshape(COV1,[sF sF supsizeU]);
    T=mtimesx(InvCOV2,COV1);
    T=bsxfun(@times,T,unitmat);
    
    T1=sum(sum(T,1),2);
    T1=T1(:);
    
    REGMEAN2=meanU(:,:,1:supsizeU);
    MEANDIFFMat=bsxfun(@minus,REGMEAN2,m1);
    MEANDIFF=reshape(MEANDIFFMat,[1 sF ,supsizeU]);
    MEANDIFFTRANS=reshape(MEANDIFFMat,[sF 1 ,supsizeU]);
    U=sum(abs(MEANDIFFTRANS(:,:)))./sF;
    dist1=mtimesx(MEANDIFF,InvCOV2);
    
    T2=mtimesx(dist1,MEANDIFFTRANS);
    T2=T2(:);
    
    distS1=mtimesx(MEANDIFF,InvCOV1);
    TS2=mtimesx(distS1,MEANDIFFTRANS);
    TS2=TS2(:);
    
    
    T3=ones(supsizeU,1)*sF;
    DET2=covDetU(1:supsizeU);
    T4=(log(Det1./DET2));
    T4=T4(:);
    
    
    TS4=log(DET2./Det1)/2;
    TS4=TS4(:);
    
    DISTMAT=(T1+T2-T3-T4)/2;
    DISTMATS=(TS1+TS2-T3-TS4)/2;
    
    DISTMAT=(DISTMAT+DISTMATS);
    
    
    
    AI(i,:)=DISTMAT;
    
    
    [i supsizeFB]
    
end
AI=AI*0.5;



