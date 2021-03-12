%{
 Objective:
    Computes the mean squared error between labels and predictions.
 Arguments:
    y_true: Ground truth values.
    y_pred: The predicted values.
 Returns:
    error: Mean squared error (MSE).
%}


function error = calc_mse(y_pred,y_true)
    error = 0;
    if size(y_pred) ~= size(y_true)
        return;
    end
    error = sum(sum((y_pred-y_true).^2)) /size(y_true,1) / 2;

end