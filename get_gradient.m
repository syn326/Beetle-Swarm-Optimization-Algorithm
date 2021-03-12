function nabla_weights = get_gradient(weights,X,y_true,activations)
    [y_pred,linear_z,active_z] = calc_ann(X,weights,activations);
    nabla_weights = weights;
    nl = size(weights,2);
    for l = nl:-1:1
        nh = size(weights{l},1);
        if l == 1
            delta_curr = (weights{l+1}(:,2:nh+1)'*delta_next).* derivative(linear_z{l},char(activations{l}));
            nabla_weights{l} = delta_curr * [1 X];
            break
        elseif l == nl
            delta_curr = (y_pred-y_true)' .* derivative(linear_z{l},char(activations{l}));

        else
            delta_curr = (weights{l+1}(:,2:nh+1)'*delta_next).*  derivative(linear_z{l},char(activations{l}));

        end
        delta_next = delta_curr;
        nabla_weights{l} = delta_curr * [1 active_z{l-1}'];
    end
end
