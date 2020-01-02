function [structAug] = augTransform(imagePath)
%{
    Generate a set of augmentations for one image and return a scructure.
    The first image is the original one, in all the others colums there are
    the augmented ones
%}

%Define augmentation parameters
vectorAngle=[30, 45, 90, 180, 240, 35, 20, 15, 60]; 

%Define a structure to contain images
structAug = struct("image", cell(1, (size(vectorAngle,2)))); 

%Get image from name path
image=imread(char(imagePath)); 

%the first image of the vector is the original one
j=1; 
structAug(j).image = image; 

%generate augmented images by rotation
for i=1:size(vectorAngle,2)
    j=j+1;
    structAug(j).image = imrotate(image, vectorAngle(i));
end

end

