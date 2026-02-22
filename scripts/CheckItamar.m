%Check Itamar code

SNRVec = linspace(10,5,50);
Fc = 10e3;
Fs = 96e3;
Ts = 20;
NumSim = 100;
NumTh = 10;

ThItamar = linspace(0.1, 0.9, NumTh);
ThDemon = linspace(0.1, 0.9, NumTh);

t = linspace(0,Ts,Ts*Fs);
sig = sqrt(2)*sin(2*pi*t*Fc);

DetectVecItamar = zeros(length(SNRVec), NumSim, NumTh);
DetectVecDemon = zeros(length(SNRVec), NumSim, NumTh);
for SNRInd = 1: length(SNRVec)
    CurrentSNR = SNRVec(SNRInd);
    for SimInd = 1: NumSim
        Rx = awgn(sig, CurrentSNR,'measured','linear');
        for ThInd = 1: NumTh
            DetectVecItamar(SNRInd, SimInd, ThInd) = DetectItamar(Rx, ThItamar(ThInd));
            DetectVecDemon(SNRInd, SimInd, ThInd) = DetectDemon(Rx, ThDemon(ThInd));
        end
    end
end

