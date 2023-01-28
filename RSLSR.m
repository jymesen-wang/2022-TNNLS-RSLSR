function [W,b] = RSLSR(W0,b0,F,X,gamma,p_norm,k)
%F: labels of data; 
%X: data; 
%gamma: regularization parameter;
%k: the percent of noise.
%The online version is: https://ieeexplore.ieee.org/document/9718600
[d,n]     = size(X);
k          = round(k *size(X,2));
obj_value0 = 0;
%update D
D      = zeros(n);
error = zeros(n,1);
S      = eye(n);
for iter = 1:30

    for i = 1: size(X,2)
        error(i) = norm(W0'*X(:,i)+b0-F(:,i),2);
        D(i,i) = (p_norm/2)*1/(power(norm(W0'*X(:,i)+b0-F(:,i),2),1-p_norm/2)+eps);
    end
    %update S
    [~, idx] = sort(error,'descend');
    idx1 = idx(1:k);
    S(:,idx1)=0;
    %update W,b
    K = S*D;
    for iter1 = 1:1
        D(i,i) = (p_norm/2)*1/(power(norm(W0'*X(:,i)+b0-F(:,i),2),1-p_norm/2)+eps);
        H = eye(n) - (K*ones(n,1)*ones(1,n))/(ones(n,1)'*K*ones(n,1));
        M = H*K*H';
        W = (X*M*X'+gamma*eye(d))\(X*M*F');
        b = (F-W'*X)*K*ones(n,1)/(ones(n,1)'*K*ones(n,1));
        W0 = W;
        b0 = b;
    end
   %calculate object value
    for j=1:n
        obj_value = obj_value0 + S(j,j)*power(norm(W0'*X(:,j)+b0-F(:,j),2),p_norm);
        obj_value0 = obj_value;
    end
    object(iter)= obj_value + gamma*norm(W0,'fro')*norm(W0,'fro');
    obj_value = 0;
    obj_value0 = 0;
    
end
end

