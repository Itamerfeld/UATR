%% Corrected Transfer Entropy for a multivariate data set, multiple surrogates setting
% This version performs also surrogate data analysis, with time shift procedure

% Y: matrix N*M of the repeated original series
% tau: matrix 1*M of the optimal embedding delays for each series
% ii: index of input series
% jj: index of output series
% c: n. of quantization levels

function [cTE,CC,scTE,sCC,UPs,UPm,sUPs,sUPm,UPfs,UPfm,VLs,VLm,cu_ms]=cTEsurro(Y,ii,jj,c,tau,u,Lmax,solitoni,numsurr,comp,zerolag,custr,tausurro,stringa)

[N,M]=size(Y);

%% normalization and quantization
for m=1:M
    Yn(:,m)=Y(:,m);
    Yq(:,m)=quantization(Yn(:,m),c)-1;
end
Yq=fix(Yq);

%% original series
disp([stringa 'first CE computation...']);
[UPfs,Vtmp,VLs,Eyj,cu_ms]=CCEnu(Yq,jj,c,tau,u,Lmax,'selfplus',zerolag,comp,ii,solitoni,custr); 
UPs=min(UPfs);

disp([stringa 'second CE computation...']);
[UPfm,Vtmp,VLm]=CCEnu(Yq,jj,c,tau,u,Lmax,'mixed',zerolag,comp,ii,solitoni,custr); 
UPm=min(UPfm);

cTE=UPs-UPm;% Transfer Entropy
CC=1-UPm/UPs; % normalized TE = causal coupling according to PRE 2011 paper


%% surrogates
if numsurr>0 % many trials: perform permutation test for significance
    X=surrotimeshift(Yq(:,ii),tausurro,numsurr);
    
    for k=1:numsurr
        Yqs=Yq; Yqs(:,ii)=X(:,k); %nella colonna del segnale di input metto l'i-esimo surrogato
        disp([stringa 'surrogate ' int2str(k) ' of ' int2str(numsurr) ', second CE computation...']);
        [sUPfm,sVtmp,sVLm]=CCEnu(Yqs,jj,c,tau,u,Lmax,'mixed',zerolag,comp,ii,solitoni,custr); 
        sUPm(k)=min(sUPfm);
        scTE(k)=UPs-sUPm(k);% Transfer Entropy
        sCC(k)=1-sUPm(k)/UPs; % normalized TE = causal coupling according to PRE paper
    end
    sUPs=UPs; % in questo caso non serve fare il self anche per i surrogati (nei surro cambia solo la input series)

else % only one trial
    scTE=[]; sCC=[]; sUPs=[]; sUPm=[];
end


% dd=1;



