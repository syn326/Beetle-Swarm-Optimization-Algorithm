%{
     Objective:
        Feed Forward NN. 
     Arguments:
        x: Given input data.
        weights: Input weights of each layers.
        activations: Previous layers' activation function.
    Returns:
        y_pred: Initialized weights of each layers.
        linear_z: Linear variable z before sigmoid activation. 
        active_z: z after sigmoid activation.

%}
function [y_pred,linear_z, active_z] = calc_ann(x,weights,activations)

        z = x';
        nl = size(weights,2);
        linear_z = cell(1,nl);
        active_z = cell(1,nl);
        % calculates z before and after sigmoid activation and gives the
        for l = 1:nl
%             disp(weights{l})
            z = linear_calc(weights{l},[1 z']');
            linear_z{l} = z;
            % Activation
            z = activation_function(z,char(activations{l}));
            active_z{l} = z;
            % Save y_pred
            if l == nl
                y_pred=z';
            end
        end
%     end
end