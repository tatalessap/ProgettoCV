function [classification] = nevusLeaveoneOut(numOfRecords,numOfAugmentation, labelsVector, featuresVector)
%% Leave-one-Out
%{
    This function split the entire dataset in Training e Test set according
    the leave-one-out approach. The training set contain all the features
    extracted, except the image that will be used in the test and all its
    augmentations. The test set contain a single features vector refered to
    image to classify.
%}

for i=1:numOfRecords
    %Portions not interesting for classification, the image itself and its varying images are eliminated
    portionToExclude = (((i-1)*numOfAugmentation)+1):(((i-1)*numOfAugmentation)+numOfAugmentation);
    
    %Training test, all images but not the image-test
    trainFeatures = featuresVector;
    trainFeatures(portionToExclude,:)=[];
    
    %Training label, all images but not the image-test
    trainLabels = labelsVector;
    trainLabels(portionToExclude,:)=[];
    
    %Test Features, only the i-th image features 
    testFeature = featuresVector(((i-1)*numOfAugmentation)+1,:);
    
    %create SVM and get the classifier model 
    SVMModel = fitclinear(trainFeatures,trainLabels);  
    
    %predict the label of the test set element
    predicted = predict(SVMModel,testFeature);     
    
    classification(i) = predicted;
end

classification = classification';
end

