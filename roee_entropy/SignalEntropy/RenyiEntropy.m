function H = RenyiEntropy(x, alpha)

%entropy for discrete variables

N = length(x);

% remap variable values
val = unique(x);
kx = length(val);
dummy(val) = 0:kx-1;
x = dummy(x);

px = histcounts(x, 0:kx)/N;

idx = find(px);

H = 1/(1-alpha)*log(sum(px(idx).^alpha));