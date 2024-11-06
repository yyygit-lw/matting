
function flag=isUniformMode(x,n)
flag=1; %是否均匀模式
S=zeros(n+1,1);
m=x;  %余数
for j=n-1:-1:0 %对比特位的循环放在外层可以提升效率，这里这样写是为可读性起见
    if m<2^j
        S(j+2)=0;
    else
        S(j+2)=1;
        m=m-2^j;
    end
end
S(1)=S(n+1);
num=0;
for j=1:n
    if S(j+1)~=S(j)
        num=num+1;
    end
    if num>2
        flag=0;
        break;
    end
end