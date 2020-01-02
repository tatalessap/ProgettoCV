function [classification] = nevusLeaveoneOut_LBP(numOfRecords, labelsVector, featuresVector)
%% Leave-one-Out
%not considered the augmented images
for i=1:numOfRecords
    %Training test, all images but not the image-test
    trainFeatures = featuresVector;
    trainFeatures(i,:)=[];
    
    %Training label, all images but not the image-test
    trainLabels = labelsVector;
    trainLabels(i,:)=[];
    
    testFeature = featuresVector(i,:);
    
    SVMModel = fitcsvm(trainFeatures,trainLabels,'KernelFunction','linear','KernelScale','auto', 'Standardize',true);  %create SVM and get the classifier model 
    
    predicted = predict(SVMModel,testFeature);     %predict the data test
        
    classification(i) = predicted;
end

classification=classification';
end