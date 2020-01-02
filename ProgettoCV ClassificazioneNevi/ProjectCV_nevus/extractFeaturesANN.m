function [featuresVector] = extractFeaturesANN(imStruct, netType)
%{
    features extraction for pre-trained Neural Network (Resnet18, AlexNet,GoogLeNet)
%}
%% Variables initialization
numRecords = size(imStruct,2);

numAug = size(imStruct(1).vectorsAug,2);

imAugStruct(numRecords) = struct();


%% Network initialization
if(ismember(netType,["resnet","alexnet","googlenet"]))
    if (netType == "resnet")
        net = resnet18;
        layer = 'pool5';
    end

    if (netType == "alexnet")
        net = alexnet;
        layer = 'fc7';
        
    end
    
    if (netType == "googlenet")
        net = googlenet;
        layer = 'loss3-classifier';
    end
else
    error("invalid network name");
end

%% Image resize and augmenter to fit the ANN required dimensions
%Get the image size required by the pre-trained network (valid for both nets)
requiredInputSize = net.Layers(1).InputSize;

for i=1:numRecords
    for j=1:numAug
        imAugStruct(i).vectorsAug(j).image = augmentedImageDatastore(requiredInputSize(1:2), imStruct(i).vectorsAug(j).image, 'OutputSizeMode','centercrop');
    end
end

%% Extract features

featuresVector=[];
for i=1:numRecords
    for j=1:numAug
        featuresVector = [featuresVector; activations(net, imAugStruct(i).vectorsAug(j).image, layer,'OutputAs','rows')];
    end
end
 
end



