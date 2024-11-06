
function out=LBP(img,radius,n)
%构建LBP图像
%使用圆形邻域，等分选取n点
 
%构造移位前后lbp映射表
map=mapOfP2shiftedP(n);
 
%构建相对位置与权重阵列
dx1=zeros(n,1);
dy1=zeros(n,1);
dx2=zeros(n,1);
dy2=zeros(n,1);
w11=zeros(n,1);
w12=zeros(n,1);
w21=zeros(n,1);
w22=zeros(n,1);
for i=0:n-1
    x=radius*cos(i*2*pi/n);
    y=radius*sin(i*2*pi/n);
    dx1(i+1)=floor(x);
    dx2(i+1)=ceil(x);
    dy1(i+1)=floor(y);
    dy2(i+1)=ceil(y);
    wx=ceil(x)-x;
    wy=ceil(y)-y;
    w11(i+1)=wx*wy;
    w21(i+1)=(1-wx)*wy;
    w12(i+1)=wx*(1-wy);
    w22(i+1)=(1-wx)*(1-wy);
end
 
out=zeros(size(img,1),size(img,2));
for y=radius+1:size(img,1)-radius
    for x=radius+1:size(img,2)-radius
        curVal=img(y,x);
        num=0;
        for i=1:n
            intVal=w11(i)*img(y+dy1(i),x+dx1(i))+w21(i)*img(y+dy1(i),x+dx2(i))+...
                       w12(i)*img(y+dy2(i),x+dx1(i))+w22(i)*img(y+dy2(i),x+dx2(i));
            if intVal>curVal
                num=num+2^(i-1);
            end
        end
        out(y,x)=map(num);
    end
end
