%% Image Classifier for nevus using Deep Learning Networks and SVM
%{
    Authors: Marta Pibiri, Riccardo Balia
    In this project we used the "PH2 dermoscopic image database" which
    contain 200 images of nevus, examinated by expert dermatologists.

    The following code, perform a classification of the images, using
    pre-trained neural networks for the features extraction, and SVM
    classifier for the prediction. Dataset division split is performed 
    according the cross validation leave-one-out method.
%}

tic %Start measure process duration.

%% Initialize variables
%read and store the image names and labels from the xlsx table
PH2dataset = [readtable("PH2Dataset/PH2_dataset.xlsx","Range","A13:A213"), readtable("PH2Dataset/PH2_dataset.xlsx","Range","C13:E213")];

datasetSize=size(PH2dataset,1);          %Image number

imPath='PH2Dataset/PH2 Dataset images/'; %Path to dataset images folder
imFolder='_Dermoscopic_Image/';          %Subfolder name part for RGB images
imFormat ='.bmp';                        %Format type

imageDir = "images";                              %folder for labelled images
commonNevusDir = imageDir + '/CommonNevus';       %subfolder for "commonNevus" labelled images
melanomaNevusDir = imageDir + '/MelanomaNevus';   %subfolder for "atypicNevus" labelled images

%% Create and populate folders according labels and Copy images in the properly folder
%Create folders if doesn't exist
if (~(isfolder(imageDir)))
    mkdir(imageDir);
    if (~(isfolder(commonNevusDir)))
        mkdir(commonNevusDir);
    end
    
    if(~(isfolder(melanomaNevusDir)))
         mkdir(melanomaNevusDir);
    end
    
    for i = 1:datasetSize
         %get the image by path
         imageName = string(PH2dataset.ImageName(i));
         currentImg = imread(imPath + imageName + '/' + imageName + imFolder + imageName + imFormat);
         
         %Save the images in properly directory according the label value
         if(string(PH2dataset.CommonNevus(i))=='X')
              imwrite(currentImg, commonNevusDir+"/"+imageName+imFormat);
         end 
         if(string(PH2dataset.Melanoma(i))=='X')
              imwrite(currentImg, melanomaNevusDir+"/"+imageName+imFormat);
         end
    end
end
%% Create the Augmented Set of the images

% Create the imageDataStore using 2 labels (common and melanoma nevus)
imds_Common = imageDatastore(commonNevusDir, 'LabelSource','foldernames');
imds_Melanoma = imageDatastore(melanomaNevusDir,'LabelSource','foldernames');

imds_2Labels = imageDatastore(cat(1,imds_Common.Files,imds_Melanoma.Files));
imds_2Labels.Labels = cat(1,imds_Common.Labels,imds_Melanoma.Labels);

originalLabels = imds_2Labels.Labels;

%Augment the current set, adding transformed copies
numOfRecords = size(imds_2Labels.Files,1);

structIm(numOfRecords) = struct();

for i=1:numOfRecords
    structIm(i).vectorsAug=augTransform(char(imds_2Labels.Files(i)));
end

numOfAugmentation = size(structIm(1).vectorsAug, 2);

%create the labels vector
labelsVector = []; %labels that includes augmentations

for i=1:numOfRecords 
    for j=1:numOfAugmentation
        labelsVector = [labelsVector; originalLabels(i)];
    end
end

%% Extract Features using different Neural Networks

featuresVectorR = extractFeaturesANN(structIm, "resnet");
featuresVectorA = extractFeaturesANN(structIm, "alexnet");
featuresVectorG = extractFeaturesANN(structIm, "googlenet");

%% Classification using SVM and leave-one-out as cross correlation split

classificationR = nevusLeaveoneOut(numOfRecords, numOfAugmentation, labelsVector, featuresVectorR);
classificationA = nevusLeaveoneOut(numOfRecords, numOfAugmentation, labelsVector, featuresVectorA);
classificationG = nevusLeaveoneOut(numOfRecords, numOfAugmentation, labelsVector, featuresVectorG);

%% Voting

%create a vector containing all the classifications
classificationCollection = [classificationR, classificationA, classificationG]; %collect classifications
labels = categorical([cellstr('CommonNevus'), cellstr('MelanomaNevus')]);

%calculate a voting 
classificationVoting = voting(classificationCollection, labels);

%Retrieve misclassified images in voting Classification
misclassifiedAmount=0;
for i=1:numOfRecords
    if(classificationVoting(i) ~= originalLabels(i))
        misclassifiedAmount=misclassifiedAmount+1;
        misclassifiedIndices(misclassifiedAmount)=i;       
    end
end

%% Plot results
%{
    We show the results through the confusion matrix and the table with accuracy
%}
titles = [];
titles = [titles; "ResNet features Extraction"];
titles = [titles; "AlexNet features Extraction"];
titles = [titles; "GoogLeNet features Extraction"];
titles = [titles; "Voting"];

figure(1); 
subplot(2,2,1), [accuracyR] = resultsDate (classificationR, originalLabels, titles(1));
subplot(2,2,2), [accuracyA] = resultsDate (classificationA, originalLabels, titles(2));
subplot(2,2,3), [accuracyG] = resultsDate (classificationG, originalLabels, titles(3));
subplot(2,2,4), [accuracyV] = resultsDate (classificationVoting, originalLabels, titles(4));

%% Plot Misclassified images (voting)
figure(2);

%compute subplot size
squareRoot = (sqrt(misclassifiedAmount));
integerPart = round(squareRoot);
if(integerPart < squareRoot)
    plotGridSize(1) = integerPart;
    plotGridSize(2) = integerPart+1;
else
    plotGridSize(1:2) = integerPart;
end

%plot images
for i=1:misclassifiedAmount
    subplot(plotGridSize(1), plotGridSize(2), i);
    imshow(char(imds_2Labels.Files(misclassifiedIndices(i))));
    xlabel(char(originalLabels(misclassifiedIndices(i)))+" seen as "+char(classificationVoting(misclassifiedIndices(i))), 'FontSize',10.1);
end 

%% Plot Misclassified images (resnet)
figure(3);

%Retrieve misclassified images
misclassifiedAmount=0;
misclassifiedIndices=[];

for i=1:numOfRecords
    if(classificationR(i) ~= originalLabels(i))
        misclassifiedAmount=misclassifiedAmount+1;
        misclassifiedIndices(misclassifiedAmount)=i;    
    end
end

%compute subplot size
squareRoot = (sqrt(misclassifiedAmount));
integerPart = round(squareRoot);
if(integerPart < squareRoot)
    plotGridSize(1) = integerPart;
    plotGridSize(2) = integerPart+1;
else
    plotGridSize(1:2) = integerPart;
end

%plot images
for i=1:misclassifiedAmount
    subplot(plotGridSize(1), plotGridSize(2), i);
    imshow(char(imds_2Labels.Files(misclassifiedIndices(i))));
    xlabel(char(originalLabels(misclassifiedIndices(i)))+" seen as "+char(classificationR(misclassifiedIndices(i))), 'FontSize',10.1);
end 

%% Show the results through tables

%Calculate statistical measurement
accuracy = [accuracyR, accuracyA, accuracyG, accuracyV]';

%Generate information tables
Tclassification = table(classificationR, classificationA, classificationG, classificationVoting, originalLabels);
Taccuracy = table(titles, accuracy);

%Display Tables
disp(Tclassification)
disp(Taccuracy);

%% Create the file.txt of the results
writetable(Taccuracy, 'Accuracy.txt');

timeElapsed = toc/60; %stop Time measurement
disp(timeElapsed); 