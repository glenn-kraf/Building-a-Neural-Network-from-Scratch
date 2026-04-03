function DL_Classification_SGD

% Uses backpropagation to train a neural network with SGD

%%%%%%% DATA %%%%%%%%%%%
x1 = [0.2,0.4,0.7,0.7,0.9,0.1,0.4,0.5,0.6,0.9];
x2 = [0.6,0.8,0.8,0.3,0.6,0.2,0.5,0.7,0.2,0.4];
y = [zeros(1,5) ones(1,5)];

% Initialize weights and biases 
rng(5000);
W2 = 0.5*randn(3,2); W3 = 0.5*randn(3,3); W4 = 0.5*randn(1,3);
b2 = 0.5*randn(3,1); b3 = 0.5*randn(3,1); b4 = 0.5*randn;

% Forward and Back propagate 
eta = 0.1;                 % learning rate
Niter = 1e6;               % number of SG iterations 

%{
Learning rate sequence
eta = [zeros(1,Niter)];
a = 100;
for j =1:Niter
    eta(j) = a/(j^(0.51));
end
%}

savecost = zeros(Niter,1); % value of cost function at each iteration

for counter = 1:Niter
    k = randi(10);         % choose a training point at random
    x = [x1(k); x2(k)];
    % Forward pass
    a2 = activate(x,W2,b2);
    a3 = activate(a2,W3,b3);
    a4 = activate(a3,W4,b4);
    % Backward pass
    delta4 = a4.*(1-a4).*(a4-y(k));
    delta3 = a3.*(1-a3).*(W4'*delta4);
    delta2 = a2.*(1-a2).*(W3'*delta3);
    % Gradient step
    W2 = W2 - eta*delta2*x';
    W3 = W3 - eta*delta3*a2';
    W4 = W4 - eta*delta4*a3';
    b2 = b2 - eta*delta2;
    b3 = b3 - eta*delta3;
    b4 = b4 - eta*delta4;
    % Monitor progress
    newcost = cost(W2,W3,W4,b2,b3,b4)   % display cost to screen
    savecost(counter) = newcost;
end

figure(3)

% Show decay of cost function

save costvec
semilogy([1:1e4:Niter],savecost(1:1e4:Niter))

  function costval = cost(W2,W3,W4,b2,b3,b4)
     costvec = zeros(10,1); 
     for i = 1:10
         x =[x1(i);x2(i)];
         a2 = activate(x,W2,b2);
         a3 = activate(a2,W3,b3);
         a4 = activate(a3,W4,b4);
         costvec(i) = norm(y(i) - a4,2);
     end
     costval = norm(costvec,2)^2;
  end % of nested function

figure(4)

% Show predicted distribution
    
    function s = neunet(o,u)
        x = [o; u];
        a2 = activate(x,W2,b2);
        a3 = activate(a2,W3,b3);
        a4 = activate(a3,W4,b4);
        s = a4;        
    end
    
    xa = linspace(0,1,1000);
    ya = linspace(0,1,1000);
    D = zeros(5,5);

    for i = 1:1000
        for j = 1:1000
            D(i,j) = neunet(xa(i),ya(j));
        end
    end

    imagesc(xa,ya,D');
    %colormap(gray);
    cm = (gray(200)); colormap(cm(150:200,:));
    set(gca,'YDir','normal');
    hold on

    scale = 6;
    set(gca,'Units', 'Inches');
    set(gca, 'Position', scale*[0.25, 0.25, 1, 1]);
    xticks([0 1]);
    xticklabels({'0','1'});
    yticks([0 1]);
    axis square;
    box on 
    hold on
    plot(0.1, 0.2, '.', 'MarkerSize',scale*7, 'MarkerEdgeColor','b');
    plot(0.4, 0.5, '.', 'MarkerSize',scale*7, 'MarkerEdgeColor','b');
    plot(0.5, 0.7, '.', 'MarkerSize',scale*7, 'MarkerEdgeColor','b');
    plot(0.6, 0.2, '.', 'MarkerSize',scale*7, 'MarkerEdgeColor','b');
    plot(0.9, 0.4, '.', 'MarkerSize',scale*7, 'MarkerEdgeColor','b');
    plot(0.2, 0.6, '.', 'MarkerSize',scale*7, 'MarkerEdgeColor','r');
    plot(0.4, 0.8, '.', 'MarkerSize',scale*7, 'MarkerEdgeColor','r');
    plot(0.7, 0.8, '.', 'MarkerSize',scale*7, 'MarkerEdgeColor','r');
    plot(0.7, 0.3, '.', 'MarkerSize',scale*7, 'MarkerEdgeColor','r');
    plot(0.9, 0.6, '.', 'MarkerSize',scale*7, 'MarkerEdgeColor','r');
    

end
