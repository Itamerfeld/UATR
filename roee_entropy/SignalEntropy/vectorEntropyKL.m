function H = vectorEntropyKL(z)

% [~, p1, p2] = unique(z, 'rows');
% 
% PDF = histcounts(p2,length(p1))/length(p2);
% H = -mean(PDF.*log2(PDF));
% 
% 
% return


z = unique(z, 'rows');

eugamma = 0.57721566490153286060651209008240243104215933593992; 
[n,p] = size(z);

[~, D] = knnsearch(z,z,'K',4,'NSMethod','exhaustive');

dist = D(:,end);
n = length(dist);

H = (p*mean(log(dist)) + log(pi^(p/2)/gamma(p/2+1)) + eugamma + log(n-1))/log(2);