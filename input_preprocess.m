function [input_data, labels] = input_preprocess()
clear all;
clc;
myFolder = 'G:\Probe\Face Rec Workshop\Code\Data set';
filePattern = fullfile(myFolder, '*.jpg'); % Change to whatever pattern you need.
theFiles = dir(filePattern);
for k = 1 : length(theFiles)
  baseFileName = theFiles(k).name;
  fullFileName = fullfile(myFolder, baseFileName);
  fprintf(1, 'Now reading %s\n', fullFileName)
  % Now do whatever you want with this file name,
  % such as reading it in as an image array with imread()
  imageArray = imread(fullFileName);
  imageArray = rgb2gray(imageArray);
   
 
imageArray =imcrop(imageArray,  [170 30 310 385] );
drawnow; % Force display to update immediately.
imageArray = imresize(imageArray,[50 50]);
imshow(imageArray);  

  imageArray = reshape(imageArray,[2500,1]);
% imageArray =imcrop(imageArray );

  if k==1
      v=imageArray;
  else
      v=[v,imageArray];
  end;
  
  
  
end


label = zeros(15,195);
class = [1,10,11,12,13,14,15,2,3,4,5,6,7,8,9];    
for i = 1:15,
    vec = zeros(15,13);
    vec(class(i),:) = ones(1,13);
    label(:,(i-1)*13 + 1:i*13) = vec;
end

index = randperm(195);

input_data = v(:,index)/255;
labels = label(:,index);

save ('Dataset.mat','v');
    
end
    