function [featuresVector] = extractFeaturesLBP(imStruct)
%{
    LBP features estraction
%}
%% Variables initialization
numRecords = size(imStruct,2);
featuresVector=[];

%% Calculate features
for i=1:numRecords
    featuresVector = [featuresVector; extractLBPFeatures(rgb2gray(imStruct(i).vectorsAug(1).image),"Upright", true)];
end

end

