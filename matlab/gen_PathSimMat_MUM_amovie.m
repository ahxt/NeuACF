load amovie;

alg = 1;

%transition matrix
%tranCell = {CNormPA,CNormPC,CNormPT; RNormPA,RNormPC,RNormPT};
%tranCell2 = {PA,PC,PT};
tranCell2 = {UI IB IC IV IT IA};
%tranCell3 = {CSNormPA,CSNormPC,CSNormPT; RSNormPA,RSNormPC,RSNormPT};
%matrix information
%infoCell = {PapMap,PaperID,PaperName; AutMap,AutID,AutName; ConfMap,ConfID,ConfName; TermMap,TermID,TermName};
%record the transition matrix type
tranType = [1,2;1,3;1,4;1,5;1,6];
% 
% len = length(path);
% tempId = path(1);
% if(tempId < 0)
%     souType = tranType(round(abs(tempId)),2);
% else
%     souType = tranType(round(abs(tempId)),1);
% end
% tempId = path(len);
% if(tempId < 0)
%     tarType = tranType(round(abs(tempId)),1);
% else
%     tarType = tranType(round(abs(tempId)),2);
% end


% %author-author Similarity
[simMatUI] = PathSim(tranCell2, [1,-1]);
[simMatIU] = PathSim(tranCell2, [-1,1]);

% [simMatUI] = PathSim(tranCell2, [1,4,-4,-1]);
% [simMatIU] = PathSim(tranCell2, [4,-4]);

simMatUI( isnan(simMatUI) ) = 0;
simMatIU( isnan(simMatIU) ) = 0;

csvwrite( '../datasets/Amovie/U.UIU.pathsim.feature.all', simMatUI )
csvwrite( '../datasets/Amovie/I.IUI.pathsim.feature.all', simMatIU )

simMatUI_modify = simMatUI;
simMatIU_modify = simMatIU;
simMatUI_modify( simMatUI_modify < mean(simMatUI_modify(find(simMatUI_modify~=0))) ) = 0;
simMatIU_modify( simMatIU_modify < mean(simMatIU_modify( find( simMatIU_modify~=0 ) )) ) = 0;
csvwrite( '../datasets/Amovie/U.UIU.pathsim.feature.mean', simMatUI_modify )
csvwrite( '../datasets/Amovie/I.IUI.pathsim.feature.mean', simMatIU_modify )


simMatUI_modify = simMatUI;
simMatIU_modify = simMatIU;
simMatUI_modify( simMatUI_modify < median(simMatUI_modify(find(simMatUI_modify~=0))) ) = 0;
simMatIU_modify( simMatIU_modify < median(simMatIU_modify( find( simMatIU_modify~=0 ) )) ) = 0;
csvwrite( '../datasets/Amovie/U.UIU.pathsim.feature.median', simMatUI_modify )
csvwrite( '../datasets/Amovie/I.IUI.pathsim.feature.median', simMatIU_modify )
%save ../datasets/ml-100k/simMat/simMat_UIUI.mat simMatUI simMatIU
%save ../datasets/ml-100k/simMat/simMat_UU_UIU.mat simMat2



%[simMat2] = PCRW(tranCell, path);

% %% cal similarity
% switch alg
%     case 1
%         [simMat] = NodeNormPSRank2(tranCell, tranType, path, souID);
%     case 2
%         [simMat] = PathSim(tranCell2, path, souID);
%     case 3
%         [simMat] = PCRW(tranCell, path, souID);
%     case 4
%         [simMat] = PathCount(tranCell2, path, souID);
%     case 5
%         [simMat] = UnitpathSimPCRW(tranCell, path, souID);%�����ӻ��ڵ�Ԫ·�����㷨
%     case 6
%         [simMat] = UnitpathSimHete(tranCell, tranType, path, souID);%�����ӻ��ڵ�Ԫ·�����㷨
%     case 7
%         [simMat] = UnitpathSimNORMPCRW(tranCell, path, souID);%�����ӻ��ڵ�Ԫ·�����㷨
%     case 8
%         [simMat] = AVGSIM(tranCell, path, souID);
% end
        