%{
Objective:
    Derivative of activation function: 
Arguments:
    x: input_value.
    activation_type
Returns
    out: value after taking derivative. 
%}
function out = derivative(x,activation_type)
if strcmp(activation_type, 'sigmoid') == 1
%         out = (exp(-x))./((1+exp(-x)).^2);
        out = activation_function(x,'sigmoid').*(1-activation_function(x,'sigmoid'));
    elseif strcmp(activation_type, 'tanh') == 1
        out = 1 - activation_function(x,'tanh').^2;
%?????     out = 4*exp(-2*x)./(1+exp(-2*x)).^2;
    elseif strcmp(activation_type, 'relu') == 1
        out = Inf(size(x));
        for i = 1:numel(x)
            if x(i) <= 0
                out(i) = 0.0;
            else
                out(i) = 1.0;
            end
        end
    else
        error("Activation Function Not Support")
    end