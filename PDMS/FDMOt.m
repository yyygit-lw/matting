function[pareto_front_bw,C,i] = FDMO(C)
i=1;
j=size(C,1);
solution_ind = 1:size(C,1);
while(i<=j)
    cmp=i+1;
    while(cmp<=j)
        if( all(C(i,:)<=C(cmp,:))&&any(C(i,:)<C(cmp,:)) )
            solution_ind([j,cmp])=solution_ind([cmp,j]);
            C([j,cmp],:)=C([cmp,j],:);
            j=j-1;
        elseif(all(C(cmp,:)<=C(i,:))&&any(C(cmp,:)<C(i,:)))
            C([i,cmp,j],:)=C([cmp,j,i],:);
            solution_ind([i,cmp,j])=solution_ind([cmp,j,i]);
            j=j-1;
            cmp=i+1;
        else
            cmp=cmp+1;
        end
    end
    i=i+1;
end
pareto_front_bw = false(size(C,1),1);
pareto_front_bw(solution_ind(1:i-1)) = 1;
end