function [hiddenWeights, outputWeights, error] = train(activationFunction, dActivationFunction, numberOfHiddenUnits, inputValues, targetValues, epochs, batchSize, learningRate)
% trainStochasticSquaredErrorTwoLayerPerceptron Creates a two-layer perceptron
% and trains it on the MNIST dataset.
%
% INPUT:
% activationFunction             : Activation function used in both layers.
% dActivationFunction            : Derivative of the activation
% function used in both layers.
% numberOfHiddenUnits            : Number of hidden units.
% inputValues                    : Input values for training (784 x 60000)
% targetValues                   : Target values for training (10 x 60000)
% epochs                         : Number of epochs to train.
% batchSize                      : Plot error after batchSize images.
% learningRate                   : Learning rate to apply.
%
% OUTPUT:
% hiddenWei.ghts                  : Weights of the hidden layer.
% outputWeights                  : Weights of the output layer.
% 

    % The number of training vectors.
    trainingSetSize = size(inputValues, 2);%210
    
    % Input vector has 784 dimensions.
    inputDimensions = size(inputValues, 1);%10000
    % We have to distinguish 10 digits.
    outputDimensions = size(targetValues, 1);%15
    
    % Initialize normal weights(samples from gaussian curve of mean=0,std=1) for the hidden layer and the output layer.
    hiddenWeights = randn(numberOfHiddenUnits, inputDimensions);%300x10000
    
    outputWeights = randn(outputDimensions, numberOfHiddenUnits);%15x300
    
    hiddenWeights = hiddenWeights./size(hiddenWeights, 2);%dividing each element by 784
    outputWeights = outputWeights./size(outputWeights, 2);%dividing each element by 300
    
    
    figure; hold on;

    for k = 1: epochs
       % Select which input vector to train on.
       n = floor(max(rand(1)*trainingSetSize - batchSize, 1));
            
       % Propagate the input vector through the network.
       inputVector = inputValues(:, n: n + batchSize );%10000x11 
       hiddenActualInput = double(hiddenWeights)*double(inputVector) ;%300x11
       hiddenOutputVector = activationFunction(hiddenActualInput) ;%300x11
       
       outputActualInput = double(outputWeights)*double(hiddenOutputVector); %15x11
       outputVector = activationFunction(outputActualInput);%15x11
       targetVector = targetValues(:, n: n + batchSize);%15x11
            
       % Backpropagate the errors.
       outputDelta = dActivationFunction(outputActualInput).*(outputVector - targetVector); %10x101
       hiddenDelta = dActivationFunction(hiddenActualInput).*(outputWeights'*outputDelta); %10x101
                   
       outputWeights = double(outputWeights) - double(learningRate).*double(outputDelta)*double(hiddenOutputVector') / batchSize;
       hiddenWeights = double(hiddenWeights) - double(learningRate).*double(hiddenDelta)*double(inputVector') / batchSize;
       if mod(k, 100) == 0
          result = activationFunction(double(outputWeights)*activationFunction(double(hiddenWeights)*double(inputVector))) - double(targetVector);%15x11
          error = 0;
          for i = 1: size(result,2)
              error = error+norm(result(:,i), 2);
          end;
          error = error / batchSize;
          plot(k/100, error,'*');
          title('Error vs Number of Epochs');
        end;
     end;
     
%      result = activationFunction(double(outputWeights)*activationFunction(double(hiddenWeights)*double(inputValues)));
%      [~,pred_ind] = max(result,[],1);
%      [~,act_ind] = max(targetValues,[],1);
%      accuracy = sum(pred_ind == act_ind)/length(pred_ind);
%      disp(accuracy);
end