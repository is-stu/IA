close all; clear; clc
%https://archive.ics.uci.edu/ml/machine-learning-databases/00602/

fullDS = readtable('Dry_Bean_Dataset.xlsx');
featMat = fullDS(:,1:16);
featMat = featMat(1:13600,:); %Le quitamos 11 registros

% Extraer las muestras de cada clase:
c1 = featMat(1:2027,:); % Muestras clase 1
c2 = featMat(2028:3350,:); % Muestras clase 2
c3 = featMat(3351:3873,:); % Muestras clase 3
c4 = featMat(3874:5504,:); % Muestras clase 4
c5 = featMat(5505:7433,:); % Muestras clase 5
c6 = featMat(7434:10070,:); % Muestras clase 6
c7 = featMat(10071:13600,:); % Muestras clase 7

realCL = [ones(2027,1); 2*ones(1322,1); 3*ones(522,1);
    4*ones(1630,1); 5*ones(1928,1); 6*ones(2636,1); 7*ones(3546,1)];

realCL = realCL(1:13600,1);
[usSamps, usLabels] = rus(c1, ones(2027,1),522);

auxInds = repmat((1:10)',1360,1);

% Muestras de entrenamiento/validación
accVec = zeros(1,10); % Vector de ceros para guardar accuracy

% Etapa de sintonizacion del No. de vecinos
vKnn = 1:2:31;
vGeoMean = zeros(1,length(vKnn));
tic
for i=1: length(vKnn)
    
    
    for auxInds2 = 1:10
        
        
        Tr_Idx = find(auxInds ~= auxInds2 );
        Val_Idx = find(auxInds == auxInds2 );
        % Entrenamiento
        Tr_Samples = featMat(Tr_Idx,:);
        Tr_Labels = realCL(Tr_Idx,:);
        % Validación
        Val_Samples = featMat(Val_Idx,:);
        Val_Labels = realCL(Val_Idx,:);
        
        knnModel = fitcknn(Tr_Samples,Tr_Labels, 'NumNeighbors', vKnn(1,i));  % Método KNN
        
        predLabels = predict(knnModel,Val_Samples);  % Predecir
        mc = confusionmat(Val_Labels,predLabels); % Matriz de confusión
        %         confusionchart(mc);
        accVec(1,auxInds2) = (mc(1,1) + mc(2,2) + mc(3,3) + mc(4,4) + mc(5,5) ...
            + mc(6,6) + mc(7,7))/sum(mc(:));
    end
    
    %figure,plot(1:10,accVec);grid on;xlabel('No. partition');ylabel('Accuracy');
    %ylim([0 1]);
    
    %     mean(accVec) % Media aritmetica / Promedio
    %     geomean(accVec) % Media geometrica
    vGeoMean(1,i) = geomean(accVec);
end
toc
figure, plot (vKnn, vGeoMean);grid on;xlabel('No. de vecinos');ylabel('Media geometrica');
Comparison = [Val_Labels, predLabels];




% ¿Cómo se hallaron las etiquetas de clase para convertirlas a
% realCL = fullDS(:,17);
%	c1 = find(strcmp('SEKER',realCL{:,:}));
%	c2 = find(strcmp('BARBUNYA',realCL{:,:}));
%	c3 = find(strcmp('BOMBAY',realCL{:,:}));