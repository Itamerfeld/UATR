%% Computation of the Conditional Entropy
% Computes the conditional entropy of the first column of B conditioned to the remaining columns (A)

function [ce,f]=CondEnt_fast(B,solitoni,cu)

% clear;close all;clc;
% B=[6 1 1; 0 1 2; 1 1 2; 74 1 1; 2 1 2; 4 3 4; 4 5 5; 3 1 2; 4 1 2; 3 3 4; 4 1 1]; % all different
% % B=[4 1 1; 0 1 2; 0 1 2; 4 1 1; 0 1 2; 3 3 4; 4 5 5; 0 1 2; 0 1 2; 3 3 4; 4 1 1]; % all the same
% % B=[4 1 1; 3 1 2; 1 1 2; 4 1 1; 0 1 2; 3 3 4; 4 5 5; 0 1 2; 1 1 2; 3 3 4; 8 1 1; 3 1 2]; %mixed

yj=B(:,1);
A=B; A(:,1)=[];

%% conto i pattern uguali dentro A (matrice ridotta) e B (matrice estesa)
[n,M]=size(A);
Q=A;
cnt=[];cnt2=[];
while ~isempty(Q)
    cmp=[];
    for m=1:M
        cmp=[cmp Q(:,m)==Q(1,m)];
    end
    tmp=(sum(cmp,2)==M);
    cnt=[cnt; sum(tmp)];
    cnt2=[cnt2; countmember(unique(yj(tmp)),yj(tmp))];
    Q(tmp,:)=[]; yj(tmp,:)=[];
end


%% CONTEGGIO SOLITONI (commentare una delle due parti qui sotto!)
% qualisolitoni: 'P' Porta way: guarda dentro i mixed pattern; 'F' Faes way: guarda dentro i self patterns

if solitoni(1)=='f'
    %% conto i solitoni MIO MODO: è solitone se è singolo dentro A
    f=0;
    for i=1:length(cnt)
       %if cnt(i)==1
       %% nuovo calcolo: invece di contare se sono singoli, conto se sono
       %% in numero <=cu (tiene conto della ridondanza/sovracampionamento)
       %% NB: torna a essere quello classico (conta i solitoni) se cu=1
       if cnt(i)>=1 &  cnt(i)<=cu 
          f=f+1;
       end
    end
    t=f;
    f=f/n;
end

if solitoni(1)=='p'
    %% conto i solitoni PORTA MODO: è solitone se è singolo dentro B
    f=0;
    for i=1:length(cnt2)
       %if cnt2(i)==1
       if cnt2(i)>=1 &  cnt2(i)<=cu
          f=f+1;
       end
    end
    t=f;
    f=f/n;
end

%% calcolo entropie
%%% ENTROPIA DI A
p=cnt./n;
%calcolo l'entropia
e=0;
for i=1:length(cnt)
   if p(i)== 0
   else
      e=e-p(i)*log(p(i));
%       e=e-p(i)*log2(p(i));
   end   
end

%%% ENTROPIA DI B
p2=cnt2./n;
%calcolo l'entropia
e2=0;
for i=1:length(cnt2)
   if p2(i)== 0
   else
      e2=e2-p2(i)*log(p2(i));
%       e2=e2-p2(i)*log2(p2(i));
   end   
end

%%% CONDITIONAL ENTROPY
ce=e2-e;

%% disps
% B
% [cnt cnt2]
% % [mark mark2]
% 
% e
% e2
% ce
