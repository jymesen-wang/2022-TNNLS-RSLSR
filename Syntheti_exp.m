clear;close all;
X_o = [0:0.2:8];
Y_o = 0.2*X_o+1.0+1*rand(1,size(X_o,2));
X_noise = [0.1,0.2,0.5,0.8,1.2];
Y_noise = [-6,-7,-3,-2.5,-5];
X = [X_o,X_noise];
Y = [Y_o,Y_noise];

[d,n]  = size(X);
obj_value0 = 0;gamma = 0.001;k = round(1/9*n);
[W_rig,b_rig] = ridge_regression(X,Y,gamma);
for iter = 1:20
    S       = eye(n);    
%     W0    = 0.9237%rand(1,1);
%     b0     = 0.8968%rand(1,1);
    W0    = W_rig;
    b0     = b_rig;

    for i = 1: n
        error(i) = norm(W0'*X(:,i)+b0-Y(:,i),2);
    end
    for j=1:n
        obj_value = obj_value0 + S(j,j)*power(norm(W0'*X(:,j)+b0-Y(:,j),2),2);
        obj_value0 = obj_value;
    end
    object(iter)= obj_value + gamma*norm(W0,'fro')*norm(W0,'fro');
    obj_value0 = 0;
    %update S
    [~, idx] = sort(error,'descend');
    idx1 = idx(1:k);
    S(:,idx1)=0;
    K = S;
    %update W,b
    H = eye(n) - (K*ones(n,1)*ones(n,1)')/(ones(n,1)'*K*ones(n,1));
    M = H*K*H';
    W = inv(X*M*X'+gamma*eye(d))*(X*M*Y');
    b = (Y-W'*X)*K*ones(n,1)/(ones(n,1)'*K*ones(n,1));
    W0 = W;
    b0  = b;
end

sz=60;
figure();stem(1:size(S,1),diag(S),'fill','m');xlabel('No. of outliers');ylabel('The value of weight');
set(gca,'FontName','Times New Roman','FontSize',15);

figure();scatter(X_o,Y_o,50,[0 0 0],'filled','b');xlabel('X');ylabel('Y'); hold on;
scatter(X_noise,Y_noise,sz,[0 0 0],'filled','mh');
legend('Original data','Outliers','location','best');set(gca,'FontName','Times New Roman','FontSize',15);

figure();
scatter(X_o,Y_o,50,[0 0 0],'filled','b');xlabel('X');ylabel('Y'); hold on;
scatter(X_noise,Y_noise,sz,[0 0 0],'filled','mh');
xlabel('X');ylabel('Y');
Y_rig = W_rig*X+b_rig;scatter(X,Y_rig,sz,[0 0 0],'filled','cs');
Y_pnlsr=W*X+b;scatter(X,Y_pnlsr,sz,[0 0 0],'filled','rd');
legend('Original data','Outliers','Ridge regression','Our method','location','best'); set(gca,'FontName','Times New Roman','FontSize',15);
