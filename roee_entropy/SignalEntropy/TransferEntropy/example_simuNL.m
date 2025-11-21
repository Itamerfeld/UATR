%% testa la cTE su simulazioni Non lineari (es: 2 Logistic unidirectionally coupled maps)
% example of Eq. (11) in L Faes, G Nollo, A Porta: 'Information-based detection of nonlinear Granger causality in multivariate processes via a nonuniform embedding technique', Phys Rev E; 2011; 83(5 Pt 1):051112.

clear;close all;clc;

%% Simulation parameters
C=0.2; %coupling strength
gamma=0.1; % percentage of additive noise
r1=3.86; r2=4; %Logistic parameters
N=500; %series length
Nt=100000; %transient


%% TE parameters
ii=1; % index of input series
jj=2; % index of output series
c=6; % n. of quantization levels
Lmax=10; %max number of candidates

soglia=-1; % decorrelation threshold: -1 means tau=1 for all channels; a common choice is 1/exp(1)
% soglia=1/exp(1); maxlags=20;

comp='n'; %compensation for instantaneous mixing
zerolag=[]; %allowed instantaneous effects , e.g. zerolag=[ [1 2]']
zerolag=[ [1 2]'];
u=1; % imposed propagation time (in nsamples, relevant to fc)
custr='n'; % oversampling correction

numsurr=10; % number of surrogates generated (0 if you don't want to generate any surrogate)
tausurro=20; % minimum shift for time-shifted sorrogates

%% Simulation generation (coupled logistic maps, Eq. 11 of PRE 2011)
y1=zeros(N+Nt,1);y2=zeros(N+Nt,1);
randn('state',sum(100*clock));
y1(1)=rand; y2(1)=rand;
for n=2:N+Nt
    y1(n)=r1*y1(n-1)*(1-y1(n-1));
    y2(n)=C*y1(n-1) + (1-C)*(r2*y2(n-1)*(1-y2(n-1)));
end

Yo=[y1(Nt+1:Nt+N) y2(Nt+1:Nt+N)];
M=2;
%%% additive noise
W=randn(N,M);
for m=1:M
    Y(:,m)=Yo(:,m)+gamma*std(Yo(:,m))*(W(:,m)-mean(W(:,m)))/std(W(:,m));
end


%% ANALYSIS
%%% evaluation of embedding delays (for each series)
if soglia==-1
    tau=ones(1,M); % take all delays identically equal to 1
else
    for i=1:M
        Z=Y(:,i);
        [zc,lags]=xcov(Z,maxlags);
        zcc=zc./zc(maxlags+1); % normalized autocorrelation
        zCC(:,i)=zcc(maxlags+1:2*maxlags+1);
        k=1;
        while k<maxlags+1
            if zCC(k,i)>soglia && zCC(k+1,i)<=soglia
                ztau=k;
                break;
            else
                k=k+1;
            end
        end
        tau(i)=ztau;
    end
end

%%% Call the main function:
[cTE,CC,scTE,sCC,UPs,UPm,sUPs,sUPm,UPfs,UPfm,VLs,VLm]=cTEsurro(Y,ii,jj,c,tau,u,Lmax,'faes',numsurr,comp,zerolag,custr,tausurro,[]);


%% Visualization
t=(1:length(Y))';
figure(1);  
for m=1:M
    subplot(M,1,m); plot(t,Yo(:,m),'k.-');hold on; plot(t,Y(:,m),'r.-');
end
    
% plot cTE analysis
Ls=size(VLs,1); Lm=size(VLm,1);
Laxis=max( [max(Ls) max(Lm)] )+1;
ymax=1.1*max(max(UPfs(:,1)),max(UPfm(:,1)));
figure(2); subplot(1,2,1); 
plot(UPfs(:,1),'k.-');hold on; plot(UPfm(:,1),'r.-');
axis([0 Laxis+1 0 ymax]);
title(['TE_' int2str(ii) '_\rightarrow_' int2str(jj) '=' num2str(cTE)],'Color','r');
for k=1:Ls(1)
    text(0.2,ymax*(1-k*0.05),['y_' num2str(VLs(k,1)) '(n-' num2str(VLs(k,2)) ')'],'FontSize',8);
end
text(0.2,ymax*(1-(k+1)*0.05),['UP_s=' num2str(UPs)],'FontSize',8);
for k=1:Lm(1)
    text(Laxis,ymax*(1-k*0.05),['y_' num2str(VLm(k,1)) '(n-' num2str(VLm(k,2)) ')'],'FontSize',8,'Color','r');
end
text(Laxis,ymax*(1-(k+1)*0.05),['UP_m=' num2str(UPm)],'FontSize',8,'Color','r');
    

if numsurr>0
    cTEth=prctile(scTE',95);
    if cTE>cTEth, strsign='significant'; else strsign='non-significant'; end

    figure(2); subplot(1,2,2);
    plot(0.8,cTE,'ro'); hold on; plot(1.2*ones(numsurr,1),scTE','co');
    line([0.8 1.2], [cTEth cTEth], 'Color','c');
    xlim([0.5 1.5]); title(strsign,'Color','r');
end



