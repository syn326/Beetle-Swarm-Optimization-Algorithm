%{ 
    Objective:  
        Calculates gamma by mutliplying the wieghts and neurons for a layer. 
    Arguments:
        weight: Input weights
        neuron_value: Current layer's value
    Returns:
        gamma = next hidden layer 
%}
function gamma = linear_calc(weight, neuron_value)

    gamma = weight * neuron_value;    
end