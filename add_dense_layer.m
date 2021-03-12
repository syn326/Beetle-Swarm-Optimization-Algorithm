%{
     Objective: 
        Helper function for initialize the model. 
        Only initializes your regular densely-connected NN layer
     Arguments:
        units: Positive integer, dimensionality of the output space.
        input_shape: the input shape.
        activation: Activation function to use.
        weights: initialize weights of each layers.
        activations: previous layers' activation function.
    Returns:
        weights: initialized weights of each layers.
        activations: each layers' activation function. 

%}
function [weights,activations] = add_dense_layer(units, input_shape, activation, weights,activations)
    % Initialize the first layer's weight by input_shape.
    if strcmp(char(input_shape),'None') == 0
        weights{end+1} = ones(units,input_shape+1);
    % If not the first layer,set input shape to 'None' and get the...
    % weight shape by previous layer's weight.
    else
        weights{end+1} = ones(units,size(weights{end},1)+1);
    end
    % Save all layers' activation funciton type.
    activations{end+1} = {activation};
    
end