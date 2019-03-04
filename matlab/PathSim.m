function [ simMat ] = PathSim(tranCell, path, beginId, endId)
%function [ simMat ] = PathSim();
%function [ simMat ] = PathSim()
%Calculate path-constrained similarity rank by Sun's work
%   transCell is the adjcent matrix cell. {transMat1, transMat2, ...}
%   beginId is the id of begin node
%   path is a row vector. i means transCell[i]; -i means transCell[i]'
%   simMat is the simliarity matrix. 

% load DBLP-DBIS-test;
% rMat1 = full(PA);
% rMat2 = full(PC);
% rMat3 = full(PT);
% tranCell = {rMat1,rMat2,rMat3};
% tranType = [1,2;1,3;1,4];
% path = [-1,1];

len = length(path);

ind = path(1);
simMat = cell2mat(tranCell(round(abs(ind))));
if(ind<0)
    simMat = simMat';
end

for i = 2:len
    ind = path(i);
    adjMat = cell2mat(tranCell(round(abs(ind))));
    if(ind<0)
        adjMat = adjMat';
    end
    simMat = simMat*adjMat;    
end
simMat = full(simMat);
clear adjMat;

tNum = size(simMat,1);
mid = round(tNum/2);
diagMat = diag(simMat);
save tempMat.mat simMat;
clear simMat;

% %%subblock operation
% tNum = size(diagMat,1);
% mid = round(tNum/2);
% AA = repmat(diagMat(1:mid)',mid,1) + repmat(diagMat(1:mid),1,mid);
% save tempMat.mat AA -append;
% clear AA;
% BB = repmat(diagMat(mid+1:tNum)',tNum-mid,1) + repmat(diagMat(mid+1:tNum),1,tNum-mid);
% save tempMat.mat BB -append;
% clear BB;
% AB = repmat(diagMat(1:mid),1,tNum-mid) + repmat(diagMat(mid+1:tNum)',mid,1);
% save tempMat.mat AB -append;
% clear AB;
% BA = repmat(diagMat(mid+1:tNum),1,mid) + repmat(diagMat(1:mid)',tNum-mid,1);
% save tempMat.mat BA -append;
% clear BA;
% %%

%%
tA = repmat(diagMat',tNum,1);
tB = repmat(diagMat,1,tNum); 
tA = tA + tB;
clear diagMat tB;
sumMat = tA;
clear tA;
%%

load tempMat.mat;
if(nargin > 2)
    simMat = simMat(beginId,:);
    sumMat = sumMat(beginId,:);
end
if nargin > 3
    simMat = simMat(endId);
    sumMat = sumMat(endId);
end

if(nargin >2)
    simMat = 2*simMat./sumMat;
else
    simBlock = mat2cell(simMat,[mid,tNum-mid],[mid,tNum-mid]);
    clear simMat;
    sumBlock = mat2cell(sumMat,[mid,tNum-mid],[mid,tNum-mid]);
    clear sumMat;

    for i = 1:2
        for j = 1:2
            pSim{i,j} = 2*simBlock{i,j}./sumBlock{i,j};
            clear simBlock{i,j} sumBlock{i,j};
        end
    end
    clear simBlock sumBlock;
    simMat = cell2mat(pSim);
    
end
end

