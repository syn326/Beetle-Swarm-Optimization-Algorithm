%{
Objective:
    activation functions
Arguments:
    linear_var. 
    activation_type
Returns:
    nolinear value(s).
%}
function nonlinear_var = activation_function(linear_var,activation_type)
    if strcmp(activation_type, 'sigmoid') == 1
        nonlinear_var = 1 ./ (1 + exp(-linear_var));
    elseif strcmp(activation_type, 'tanh') == 1
        nonlinear_var = 1 - 2 ./ (1 + exp(2*linear_var));
%  ????        nonlinear_var = 2 ./ (1 + exp(-2*linear_var));
%         nonlinear_var = tanh(linear_var);
    elseif strcmp(activation_type, 'relu') == 1
        nonlinear_var = max(0.0,linear_var);
    else
        error("Activation Function Not Support")
    end
    
end