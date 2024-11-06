% This function solves the proposed trace minimization regularized by
% row-sparsity norm using an ADMM framework
% D: dissimilarity matrix
% p: norm of the mixed L1/Lp regularizer, {2,inf}
% options: parameter settings for ADMM
% C2: solution of DS3
%--------------------------------------------------------------------------
% Copyright @ Ehsan Elhamifar, 2014
%--------------------------------------------------------------------------

function C2 = ds3solver_regularized(D,p,options)

if (nargin < 2)
    p = inf;
end

if (nargin < 3)
    [rho_min, rho_max] = computeRegularizer(D,p);
    options.rho = .1 * rho_max;
    options.mu = 1 * 10^-1;
    options.maxIter = 3000;
    options.errThr = 1 * 10^-7;%1*10^-7  eski değer
    options.verbose = true;
end

rho = options.rho;
mu = options.mu;
maxIter = options.maxIter;
errThr = options.errThr;
verbose = options.verbose;
CFD = ones(size(D,1),1);
ratio = 0.1;

% initialization
[Nr,Nc] = size(D);
terminate = false;
k = 1;
[~,idx] = min(sum(D,2));
C1 = zeros(size(D));
C1(idx,:) = 1;
Lambda = zeros(Nr,Nc);

% running iterations
while (~terminate)
    Z = shrinkL1Lp(C1-(Lambda+D)./mu,rho/mu*CFD,p);
    C2  = solver_BCLS_closedForm(Z + Lambda ./ mu);
    
    Lambda = Lambda + mu .* (Z - C2);
    
    err1 = errorCoef(Z,C2);
    err2 = errorCoef(C1,C2);
    repNum=length(findRepresentatives(C2));
     if (repNum<2) % Ben ekledim.
        [rho_min, rho_max] = computeRegularizer(D,p);
        alpha=options.rho/rho_max;
%         repNum
%         alpha
        options.verbose = verbose;
        options.rho = (alpha*0.8) * rho_max; % regularization parameter
        options.mu = 1 * 10^-1;
        options.maxIter = 3000;
        options.errThr = 1 * 10^-7;
        rho = options.rho;
        mu = options.mu;
        maxIter = options.maxIter;
        errThr = options.errThr;
        verbose = options.verbose;
        CFD = ones(size(D,1),1);
        ratio = 0.1;
        
        % initialization
        [Nr,Nc] = size(D);
        terminate = false;
        k = 1;
        [~,idx] = min(sum(D,2));
        C1 = zeros(size(D));
        C1(idx,:) = 1;
        Lambda = zeros(Nr,Nc);% Buraya kadar alttaki elseif if olacak
   
    elseif ( k >= maxIter || (err1 <= errThr && err2 <= errThr) )
        terminate = true;
        if (verbose)
            fprintf('Terminating: \n');
            fprintf('||Z-C||= %1.2e, ||C1-C2||= %1.2e, repNum = %3.0f, iteration = %.0f \n\n',err1,err2,repNum,k);
        end
%          if (repNum>150)
%               [rho_min, rho_max] = computeRegularizer(D,p);
%         alpha=options.rho/rho_max;
%         options.verbose = verbose;
%         options.rho = (alpha*1.2) * rho_max; % regularization parameter
%         options.mu = 1 * 10^-1;
%         options.maxIter = 3000;
%         options.errThr = 1 * 10^-7;
%         rho = options.rho;
%         mu = options.mu;
%         maxIter = options.maxIter;
%         errThr = options.errThr;
%         verbose = options.verbose;
%         CFD = ones(size(D,1),1);
%         ratio = 0.1;
%         
%         % initialization
%         [Nr,Nc] = size(D);
%         terminate = false;
%         k = 1;
%         [~,idx] = min(sum(D,2));
%         C1 = zeros(size(D));
%         C1(idx,:) = 1;
%         Lambda = zeros(Nr,Nc);% Buraya kadar alttaki elseif if olacak
%         terminate=false;
%         k=1;
%          end


    else
        k = k + 1;
        if (verbose)
            if (mod(k,100)==0)
                fprintf('||Z-C||= %1.2e, ||C1-C2||= %1.2e, repNum = %3.0f, iteration = %.0f \n',err1,err2,repNum,k);
            end
        end
    end
    C1 = C2;
end
