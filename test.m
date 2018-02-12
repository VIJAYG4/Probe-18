   inputImage = imread('G:\Probe\Face Rec Workshop\Code\Test\14-13.jpg');
   inputImage = rgb2gray(inputImage);
   inputImage =imcrop(inputImage ,  [170 30 310 385] );
drawnow; % Force display to update immediately.
inputImage = imresize(inputImage,[50 50]);
inputImage = reshape(inputImage,[2500,1]);

inputImage = inputImage/255;
imshow(inputImage);  
  
     result = activationFunction(double(outputWeights)*activationFunction(double(hiddenWeights)*double(inputImage)));
     [~,pred_ind] = max(result,[],1);
     disp(result);
     disp(pred_ind);
     
     % Wrong Classes: 2,5,13,14