%Run entropy check

addpath('/Users/roee/Dropbox (University of Haifa)/BiomimikInterceptor/SignalEntropy')
addpath('/Users/roee/Dropbox (University of Haifa)/BiomimikInterceptor/SignalEntropy/TransferEntropy');
addpath('/Users/roee/Dropbox (University of Haifa)/PinkDolphins');


%% parameters
Quantize = 100e3;
RxFs = 192e3;
MaxSamples = 10e3;
%for approximate entropty
DistVal = 0.5;
WinLen = 5;

%for Renyi entropy
alpha = 0.5;

%for sample entropy
dist_type = 'chebychev';
Dim = 1;
Tolerance = 0.5;

%for transfer entropy
tau=ones(1,2);
QunatizationLevel = 6;
PropagationTime = 1;
MaxCandidates = 10;
NumSurogateGen = 0;
comp='n'; %compensation for instantaneous mixing
allowed_instantaneous_effects = [1 2]';
custr='n'; % oversampling correction
tausurro=0; % minimum shift for time-shifted sorrogates

%for Tsallis entropy
alphaTsallis = 0.1;
NormTsallis = 2;

%% received signals
MaxFile = 4000;
MaxBlock = 100;

DiffEnt = zeros(2, MaxFile, MaxBlock);
MutalEnt = zeros(2, MaxFile, MaxBlock);
TransferEnt = zeros(2, MaxFile, MaxBlock);
TransferNormEnt = zeros(2, MaxFile, MaxBlock);
VectorEnt = zeros(2, MaxFile, MaxBlock);
VectorEnt2 = zeros(2, MaxFile, MaxBlock);
ApproxEnt = zeros(2, MaxFile, MaxBlock);
RenyiEnt = zeros(2, MaxFile, MaxBlock);
SampleEnt = zeros(2, MaxFile, MaxBlock);
TsallisEnt = zeros(2, MaxFile, MaxBlock);

DiffEntB = zeros(2, MaxFile);
MutalEntB = zeros(2, MaxFile);
TransferEntB = zeros(2, MaxFile);
TransferNormEntB = zeros(2, MaxFile);
VectorEntB = zeros(2, MaxFile);
VectorEnt2B = zeros(2, MaxFile);
ApproxEntB = zeros(2, MaxFile);
RenyiEntB = zeros(2, MaxFile);
SampleEntB = zeros(2, MaxFile);
TsallisEntB = zeros(2, MaxFile);

for AnalyzeID = 1: 2
    if AnalyzeID == 1
        cd('Signal');
    else
        cd('Noise');
    end
    d = dir;

    for FileIndexer = 3: length(d)
        FileIndexer
        if FileIndexer <= length(d)
            CurrentFile = d(FileIndexer).name;
            if strfind(CurrentFile, 'mat')
                load(CurrentFile);
                Sig_seg = sig;
                Fs = fs;

                L = length(Sig_seg);

                %reference signal
                Amp = max(Sig_seg) - min(Sig_seg);
                RefSig = randn(L, 1) * Amp;
                RefSig = RefSig - min(RefSig);
                RefSig = round(RefSig/max(RefSig)*Quantize)+1;

                %convert to positive integents
                Sig_seg = Sig_seg - min(Sig_seg);
                Sig_seg = round(Sig_seg/max(Sig_seg)*Quantize)+1;

                NumBlock = ceil(L / MaxSamples);
                E1 = zeros(1, NumBlock);
                E2 = zeros(1, NumBlock);
                E3 = zeros(1, NumBlock);
                E4 = zeros(1, NumBlock);
                E5 = zeros(1, NumBlock);
                E6 = zeros(1, NumBlock);
                E7 = zeros(1, NumBlock);
                E8 = zeros(1, NumBlock);
                E9 = zeros(1, NumBlock);
                E10 = zeros(1, NumBlock);

                CurrentSample = 1;
                for BlockInd = 1: NumBlock
                    CurrentVec = CurrentSample: min([CurrentSample + MaxSamples - 1, L]);
                    CurrentSample = CurrentSample + MaxSamples;

                    E1(BlockInd) = DifferentialEntropy(Sig_seg(CurrentVec), RefSig(CurrentVec));
                    E2(BlockInd) = MutualInformation(Sig_seg(CurrentVec), RefSig(CurrentVec));

                    Sig = [RefSig(CurrentVec), Sig_seg(CurrentVec)];
                    [cTE,CC,scTE,sCC,UPs,UPm,sUPs,sUPm,UPfs,UPfm,VLs,VLm]=cTEsurro(Sig,1,2,QunatizationLevel,tau,PropagationTime,MaxCandidates,'faes',NumSurogateGen,comp,allowed_instantaneous_effects,custr,tausurro,[]);
                    E3(BlockInd) = cTE;
                    E4(BlockInd) = CC;

                    E5(BlockInd) = vectorEntropyKL(Sig);
                    Sig2 = [RefSig(CurrentVec), Sig_seg(CurrentVec)];
                    VecEnt2 = vectorEntropyKL(Sig2);
                    E6(BlockInd) = E5(BlockInd) - VecEnt2;

                    E7(BlockInd) = ApproximateEntropy(WinLen,DistVal,Sig_seg(CurrentVec));
                    E8(BlockInd) = RenyiEntropy(Sig_seg(CurrentVec), alpha);
                    E9(BlockInd) = SampleEntropy(Sig_seg(CurrentVec), Dim, Tolerance, dist_type);
                    E10(BlockInd) = TsallisEntropy(Sig_seg(CurrentVec), alphaTsallis, NormTsallis);
                end
                FileIndexer
                DiffEnt(AnalyzeID, FileIndexer-2, 1: length(E1)) = (E1);
                MutalEnt(AnalyzeID, FileIndexer-2, 1: length(E2)) = (E2);
                TransferEnt(AnalyzeID, FileIndexer-2, 1: length(E3)) = (E3);
                TransferNormEnt(AnalyzeID, FileIndexer-2, 1: length(E4)) = (E4);
                VectorEnt(AnalyzeID, FileIndexer-2, 1: length(E5)) = (E5);
                VectorEnt2(AnalyzeID, FileIndexer-2, 1: length(E6)) = (E6);
                ApproxEnt(AnalyzeID, FileIndexer-2, 1: length(E7)) = (E7);
                RenyiEnt(AnalyzeID, FileIndexer-2, 1: length(E8)) = (E8);
                SampleEnt(AnalyzeID, FileIndexer-2, 1: length(E9)) = (E9);
                TsallisEnt(AnalyzeID, FileIndexer-2, 1: length(E10)) = (E10);

                DiffEntB(AnalyzeID, FileIndexer-2) = mean(E1);
                MutalEntB(AnalyzeID, FileIndexer-2) = mean(E2);
                TransferEntB(AnalyzeID, FileIndexer-2) = mean(E3);
                TransferNormEntB(AnalyzeID, FileIndexer-2) = mean(E4);
                VectorEntB(AnalyzeID, FileIndexer-2) = mean(E5);
                VectorEnt2B(AnalyzeID, FileIndexer-2) = mean(E6);
                ApproxEntB(AnalyzeID, FileIndexer-2) = mean(E7);
                RenyiEntB(AnalyzeID, FileIndexer-2) = mean(E8);
                SampleEntB(AnalyzeID, FileIndexer-2) = mean(E9);
                TsallisEntB(AnalyzeID, FileIndexer-2) = mean(E10);

            end
        end
    end
    cd ..
    if AnalyzeID == 1
        save('PinkDolphinResultsSig', 'DiffEnt', 'MutalEnt', 'TransferEnt', 'TransferNormEnt', ...
        'VectorEnt', 'VectorEnt2', 'ApproxEnt', 'RenyiEnt', 'SampleEnt', 'TsallisEnt', ...
        'DiffEntB', 'MutalEntB', 'TransferEntB', 'TransferNormEntB', ...
        'VectorEntB', 'VectorEnt2B', 'ApproxEntB', 'RenyiEntB', 'SampleEntB', 'TsallisEntB');

    else
        save('PinkDolphinResultsNoise', 'DiffEnt', 'MutalEnt', 'TransferEnt', 'TransferNormEnt', ...
        'VectorEnt', 'VectorEnt2', 'ApproxEnt', 'RenyiEnt', 'SampleEnt', 'TsallisEnt', ...
        'DiffEntB', 'MutalEntB', 'TransferEntB', 'TransferNormEntB', ...
        'VectorEntB', 'VectorEnt2B', 'ApproxEntB', 'RenyiEntB', 'SampleEntB', 'TsallisEntB');

    end
end

%% decode
%DecodeScriptHaifa;

