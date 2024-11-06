
function map=mapOfP2shiftedP(n)
%构建由移位前LBP值到最小移位LBP值的map
Key=num2cell(0:2^n-1); 
matVal=zeros(2^n,1);
NoneUniform=0;
for i=0:n-1
    isEven=mod(i,2);
    NoneUniform=NoneUniform+isEven*2^i; %构造一个非均匀模式
end
for i=0:2^n-1
    val=zeros(n,1);
    val(n)=i;
    % 求移位后的取值，断点从高位往低位移动
    for j=n-1:-1:1
        p=0; %高位部分
        q=0; %低位部分
        if i<2^j
            q=i;
            p=0;
        else
            q=mod(i,2^j);
            p=(i-q)/2^j;
        end
        val(j)=q*2^(n-j)+p;
    end
    minVal=min(val);
    if isUniformMode(minVal,n) 
        matVal(i+1)=minVal;
    else
        matVal(i+1)=NoneUniform; %非均匀模式统一为一种模式
    end
end
Val=num2cell(matVal);
map=containers.Map(Key,Val);
