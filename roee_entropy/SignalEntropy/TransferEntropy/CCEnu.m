%% CORRECTED CONDITIONAL ENTROPY WITH NON UNIFORM EMBEDDING performed with lag tau
%% new version: adds propagation time u, implements correction for instantaneous causality (if desired)
%%% INPUTS:
% Yq: N*M matrix of the M QUANTIZED signals each having length N (modificato 12-1-2012: la quantizzazione la faccio fuori da questa funzione, così velocizzo dato che la chiamo 2 volte)
% j: index (column) of the series considered as output, the one we want to describe
% c: n. of quantization levels
% tau: time lag of the candidates: it is a vector of M lag values (one for each series)
% u: propagation time lag for series other than the output series j
% Lmax: %max number candidates for each series
% candidate_set: 'self', , 'selfplus' 'mixed'
% zerolag: 
% comp: flag for possible compensation for instantaneous effects
% ii: if candidate_set==selfplus, ii is the index of the input series
% solitoni: parameter to be passed to CondEnt.m for for the CE correction (faes o porta)

function [cceOK, V, VL, Eyj, cu_m]=CCEnu(Yq,j,c,tau,u,Lmax,candidate_set,zerolag,comp,ii,solitoni,custr);
narginchk(7,12) 
if nargin<12, custr='y'; end % default correggo con ceil(cu) per sovracampionamento/ridondanza
if nargin < 11, solitoni='porta'; end %default stima cce classica(porta)
if nargin < 10, ii=[]; end %default non serve la serie di input (l'embedding non è selfplus)
if nargin < 9, comp='n'; end %default non compenso per effetti immediati
if nargin < 8, zerolag=[]; end %default non permetto effetti immediati

%% for internal test (leave commented)
% clear; close all; clc;
% c=6; % n. of quantization levels
% Lmax=4; %max number candidates for each series
% j=1; % output series (to be "predicted")
% candidate_set='selfplus'; %'selfplus' or 'mixed'
% ii=[2]; % input series (when expected depending on the kind of embedding)
% solitoni='faes';
% comp='y';
% zerolag=[ [1 2]', [1 3]' [2 3]' ];
% % zerolag=[];
% custr='si';
% 
% %%% SERIES
% %%% JUST AN EXAMPLE
% x=[1 2 0 1 5 3 3 5 4 4 3 4 0 5 1 0 0 3]';
% y=[1 1 2 1 3 4 1 3 0 0 5 4 2 2 1 5 0 1]';
% z=[3 3 4 0 5 0 2 2 3 0 3 4 4 4 5 1 2 2]';
% Yq=[x y z];
% tau=[1 1 1];
% u=5;
% 
% %%% UNIDIRECTIONALLY COUPLED AR PROCESSES
% % ro=0.95; coup=1; lung=300;
% % [x,y]=simu_ARuni(ro,coup,lung);
% 
% %%% UNIDIRECTIONALLY COUPLED HENON MAPS
% % coup=0; lung=300;
% % [x,y]=simu_henon_uniUSA(coup,lung);
% % 
% % Y=[x y];

%% 2) Candidate components (realizza lo schema di cTE study)
[~,M]=size(Yq);

switch candidate_set
    %%% SELFPLUS PREDICTION Candidati sono tutti i valori passati (fino a tau*Lmax) di tutti i segnali tranne quello di input
    case {'selfplus'}
        %candidates=NaN*ones((M-1)*Lmax,3); %col 1: indice segnale; col 2: indice lag; col 3:marker dei candidati inclusi
        candidates=[];
        mm=1;
        for m=1:M
            if m~=ii % se non input
                candidates((mm-1)*Lmax+1:mm*Lmax,1)=m*ones(Lmax,1);
                candidates((mm-1)*Lmax+1:mm*Lmax,2)=tau(m)*(1:Lmax)';
                candidates((mm-1)*Lmax+1:mm*Lmax,3)=NaN*ones(Lmax,1);
                if m~=j % se non output, inserisco il propagation time
                    candidates((mm-1)*Lmax+1:mm*Lmax,2)=candidates((mm-1)*Lmax+1:mm*Lmax,2)+u-tau(m);  
                end
                mm=mm+1;
            end
        end
        if comp=='n' % se la compensazione per effetti immediati non è attiva, 
            %in coda aggiungo gli eventuali candidati a effetti immediati (da zerolag)
            for mm=1:size(zerolag,2)
                if zerolag(2,mm)==j && zerolag(1,mm)~=ii
                    candidates=[candidates; zerolag(1,mm) 0 NaN];
                end
            end
        else % se compenso per effetti immediati
            for mm=1:M
                if mm~=j
                    candidates=[candidates;[mm 0 NaN]];
                end
            end
        end
        
    %%% MIXED PREDICTION Candidati sono tutti i valori passati (fino a Lmax) di tutti i segnali
    case {'mixed'}
        %candidates=NaN*ones(M*Lmax,3); %col 1: indice segnale; col 2: indice lag; col 3:marker dei candidati inclusi
        candidates=[];
        for m=1:M
            candidates((m-1)*Lmax+1:m*Lmax,1)=m*ones(Lmax,1);
            candidates((m-1)*Lmax+1:m*Lmax,2)=tau(m)*(1:Lmax)';
            candidates((m-1)*Lmax+1:m*Lmax,3)=NaN*ones(Lmax,1);
            if m~=j % se non output, inserisco il propagation time
              candidates((m-1)*Lmax+1:m*Lmax,2)=candidates((m-1)*Lmax+1:m*Lmax,2)+u-tau(m);  
            end
        end
        if comp=='n' % se la compensazione per effetti immediati non è attiva, 
            %in coda aggiungo gli eventuali candidati a effetti immediati (da zerolag)
            for mm=1:size(zerolag,2)
                if zerolag(2,mm)==j
                    candidates=[candidates; zerolag(1,mm) 0 NaN];
                end
            end
        else % se compenso per effetti immediati
            for mm=1:M
                if mm~=j
                    candidates=[candidates;[mm 0 NaN]];
                end
            end
        end
   
end



%% 3) test the candidates
exitcrit=0; %exit criterion: stay in loop if exitcrit==0
V=[]; k=1;

Eyj=ShannEnt(Yq(:,j)); %Shannon Entropy of yj(n)
while exitcrit==0
    
ce=NaN*ones(size(candidates,1),1); cce=ce;

for i=1:size(candidates,1)
    if isnan(candidates(i,3))%test i-th candidate, only if not already included in V
        Vtmp=[V; candidates(i,1:2)]; 
        
        % form the vectors from the signals in Y according to the candidates in Vtmp
        B=buildvectors(Yq,j,Vtmp); % new function, builds directly B with yj(n)    
        
        if custr=='y' %eventuale correzione per il sovracampionamento/ridondanza (conta i pattern uguali di seguito e fa la media)
            cu=[];q=1;
            while q+1<length(B)
                s=q;
                while q+1<length(B) & B(q,1)==B(q+1,1) & B(q,2)==B(q+1,2), q=q+1; end
                cu=[cu; q-s+1];
                q=q+1;
            end
            cu_m=ceil(mean(cu));
        else
            cu_m=1;
        end
        %%%disp(fix(mean(cu)));
        %%[ce(i),perc]=CondEnt(B,solitoni,ceil(mean(cu))); % conditional entropy if we add the i-th candidate
        [ce(i),perc]=CondEnt_fast(B,solitoni,cu_m); %new fast version
        
        %[ce(i),perc]=CondEnt(B,solitoni); % conditional entropy if we add the i-th candidate
        cce(i)=ce(i)+perc*Eyj; % corrected conditional entropy if we add the i-th candidate
        %%%%%cce(i)=ce(i);
    end
end    
    
%% 4) select the candidate and update the embedding vector    
ind_sel=find(cce==min(cce)); 
% ind_sel=find(ce==min(ce)); % misura altervativa!
if ~isempty(ind_sel)
    ind_sel=ind_sel(1); % index of the selected candidate
    candidates(ind_sel,3)=1; %mark as selected
    V=[V; candidates(ind_sel,1:2)]; %update embedding vector
    cceOK(k)=min(cce); %update the minimum cce at step k
else    % ho testato tutti i candidati ed è sempre sceso!
    if k == 1
        cceOK(k) = 0;
    else
    cceOK(k)=cceOK(k-1); %la fisso pianeggiante, per poter uscire
    V=[V; V(k-1,1:2)]; %fake update embedding vector (copio il precedente)
    end
end

%% 5) test for exit criterion
if k>1
    if cceOK(k)>=cceOK(k-1) % a minimum is found: exit
        exitcrit=1;
    else
        k=k+1;
    end
else
    if cceOK(k)==0 % cannot be more predictable: exit
        exitcrit=1;
    else
        k=k+1;
    end
end

end %endwhile
    
%% 6) after exiting: Unpredictability
cceOK=cceOK';
if k==1 % if maximum predictability is found at first step
    L=k; VL=V;
else % a minimum can be found
    L=k-1; %optimal embedding length
    VL=V(1:L,:); %optimal embedding: exclude the last one
end

CCEmin=cceOK(L);

% UP=cceOK(L)/Eyj; % Unpredictability
% UPf=cceOK./Eyj; % unpredictability function

%% disp
% figure(1); plot(UPf,'r.-'); axis([0 k+1 0 1]);
% 
% VL
% UP
% ii
% j
% candidates
