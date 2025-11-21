function H = TsallisEntropy(x, alpha, q)

%Tsallis entropy yields power-law distribution, whereas Shannon entropy yields exponential ...
%equili- brium distribution.

N = length(x);

% remap variable values
val = unique(x);
kx = length(val);
dummy(val) = 0:kx-1;
x = dummy(x);

px = histcounts(x, 0:kx)/N;

idx = find(px);

H = (1 - sum(px(idx).^alpha)) / (q-1);
