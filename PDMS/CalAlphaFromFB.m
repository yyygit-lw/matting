function [ alpha ] = CalAlphaFromFB( F,B,U )
%CALALPHAFROMFB Summary of this function goes here
%   Detailed explanation goes here
    if isa(F,'uint8')
        F = single(F);
    end
    if isa(B,'uint8')
        B = single(B);
    end
    if isa(U,'uint8')
        U = single(U);
    end
    if size(F,1) ~= size(B,1)||size(F,1) ~= size(U,1)
        error('F,B,U must have the same rows');
    end
    if size(F,2) ~= size(B,2)
        error('F,B must have the same cols');
    end
    if size(F,2) ~= size(U,2)
        if size(U,2) == 1
            U = repmat(U,1,size(F,2));
        else
            error('F,B must have the same cols');
        end
    end
    alpha = sum((U - B).* (F - B))./( (sum((F - B).^2))+0.0001) ;
    

end

