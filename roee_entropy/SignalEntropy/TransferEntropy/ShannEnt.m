%% Computation of the Shannon Entropy
% Computes the shannon entropy of the vectors collected as rows in the matrix A

function [e,f]=ShannEnt(A)

% clear;close all;clc;
% % A=[6 1 1; 0 1 2; 1 1 2; 74 1 1; 2 1 2; 4 3 4; 4 5 5; 3 1 2; 4 1 2; 3 3 4; 4 1 1]; % all different
% A=[4 1 1; 0 1 2; 0 1 2; 4 1 1; 0 1 2; 3 3 4; 4 5 5; 0 1 2; 0 1 2; 3 3 4; 4 1 1]; % some same

%% conto i pattern uguali dentro A
n=size(A,1);
mark=zeros(n,1);
cnt=zeros(n,1);
for i=1:n
   if mark(i)==0
      cnt(i)=cnt(i)+1;
      mark(i)=mark(i)+1;
      for j=i+1:n
         if A(j,:)==A(i,:)
            cnt(i)=cnt(i)+1;
            mark(j)=mark(j)+1;
         end
      end
   end
end

%% conto i solitoni
f=0;
for i=1:n
   if cnt(i)==1
      f=f+1;
   end
end
t=f;
f=f/n;

%% calcolo entropia
p=cnt./n;
e=0;
for i=1:n
   if p(i)== 0
   else
      e=e-p(i)*log(p(i));
   end   
end


%% disps
% A
% [cnt mark]
% e
