% genera x surrogato con time shift
% serve per studiare la significatività di correlazioni short term con shift su distanze long term - preserva il massimo della struttura della singola serie

function xs=surrotimeshift(x,taumin,numsurr)

error(nargchk(1,3,nargin));%min e max di input arguments
if nargin < 3, numsurr=1; end %default 1 sola serie surrogata
if nargin < 2, taumin=1; end %default cerco a partire dallo shift di 1 solo sample

% percorso='D:\johnny\lavoro\integrate_nlpred\elaborati_loo_si\';% percorso dei dati da analizzare
% nomefile='t-ca.prn';
% rs=load([percorso nomefile]);
% x=rs(:,1); 
% x=(1:15)';
% numsurr=100;
% taumin=4;

N=length(x);
taumax=N-taumin;

tauvett=randperm(taumax-taumin+1)+taumin-1;% vettore di tutti i possibili shift fra taumin e taumax

if length(tauvett)<numsurr
    error('il numero di shift concessi è minore del numero di surrogati che vuoi generare');
end

xs=zeros(N,numsurr);
for i=1:numsurr
    xs(:,i)=circshift(x,tauvett(i));
end

