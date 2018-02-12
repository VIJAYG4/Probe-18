
     result = activationFunction(double(outputWeights)*activationFunction(double(hiddenWeights)*double(inputValues)));
     [~,pred_ind] = max(result,[],1);
     [~,act_ind] = max(targetValues,[],1);
     accuracy = sum(pred_ind == act_ind)/length(pred_ind);
     disp(accuracy);