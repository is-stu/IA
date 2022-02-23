%funcion
function [usSamps, usLabels] = rus(inSamps, inLabels,n)
%RUS: Random Under Sampling/submuestreo aleatorio
ssN = randperm(size(inSamps,1)); %genera la cantidad de numeros random que se le pase.
usSamps = inSamps(ssN(1:n),:);
usLabels = inLabels(ssN(1:n),1);
end