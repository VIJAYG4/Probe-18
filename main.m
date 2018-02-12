%applyStochasticSquaredErrorTwoLayerPerceptronMNIST Train the two-layer
%perceptron using the FEI dataset and evaluate its performance.


%Training Face database

% Load FEI dataset.
    [inputValues, targetValues] = input_preprocess();
    % Transform the labels to correct target values.
    
    
    % Choose form of MLP:
    numberOfHiddenUnits = 1250;
    
    % Choose appropriate parameters.
    learningRate = 0.1;
    
    % Choose activation function.
    activationFunction = @logisticSigmoid;
    dActivationFunction = @dLogisticSigmoid;
    
    
    
    % Choose batch size and epochs. Remember there are 60k input values.
    batchSize = 20;
    epochs = 10000;
    
    fprintf('Train twolayer perceptron with %d hidden units.\n', numberOfHiddenUnits);
    fprintf('Learning rate: %d.\n', learningRate);
    
    
    [hiddenWeights, outputWeights, error] = train(activationFunction, dActivationFunction, numberOfHiddenUnits, inputValues, targetValues, epochs, batchSize, learningRate);
    save Train.mat hiddenWeights outputWeights
                
    
