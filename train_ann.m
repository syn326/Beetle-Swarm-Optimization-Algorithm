%{
    Objective: 
        Trains the NN
    Arguments:
        weights: Weights.
        learning_rate: Floating point value,The learning rate.
        x: Given Input data.
        d: Given Output data.
        activations: Cell array, activation type of each layer
        epochs: Max number of epochs.
        threshold: Error tolerance.
    Returns:
        y_pred: predicted y value after training.
        error: MSE error.
        weights: Updated weights.
%}

function [y_pred,error,error_history,weights] = train_ann(weights,learning_rate,x,d,activations,epochs,threshold,opt)
    count = 0;
    error_history = [];
    y_pred = zeros(size(d));
    prev_error = Inf;
    error = intmax;
    %for QN
    H_inv = eye(size(flatten_weights(weights),1));
    c = 1;
    %Init. for BSO
    step = 1;
    n_beetles = 5;
    c1 = 0.95;
    c2 = 5;
    lambda = 0.8;
    %init. beetles
    beetle = flatten_weights(weights);
%     beetle = sign(rand(1)-rand(1))*rand(size(beetle));
    S = size(beetle,1);
    beetles = ones(S,n_beetles);
    i_best = beetles;
    g_best = beetle;
    g_err = Inf;
    weights_shape = cellfun(@size,weights,'uni',false);
    V = rand(S,n_beetles);
    V_max = 0.1;
    V_min = 0.0001;
    prev_i_err = Inf(n_beetles,1);
    %
    while ((count < epochs) && (error > threshold))
        prev_weights = weights;
        step = step * 0.95;
        for i = 1:size(x,1)
            if strcmp(opt,"dampedQN") == 1
                [weights,H_inv] = QN_update_weights(weights,H_inv,x(i,:),d(i,:),learning_rate,activations,true);
            elseif strcmp(opt,"QN") == 1
                [weights,H_inv] = QN_update_weights(weights,H_inv,x(i,:),d(i,:),learning_rate,activations,false);
            
            
            elseif strcmp(opt,"BSO") == 1
%                 gif('BSOgif.gif');
                [beetles,V] = BSO(x(i,:),d(i,:),beetles,V,weights_shape, step,n_beetles,c1,c2,V_max,V_min,lambda,count,epochs,i_best, g_best,activations);
                for j = 1:n_beetles
                    beetle_curr = pack_weights(beetles(:,j),weights_shape);
                    y_pred_curr = nan(size(d));
%                     for k = 1:size(x,1)
%                         [y_pred_sample,~,~] = calc_ann(x(k,:),beetle_curr,activations);
%                         y_pred_curr(k,:) = y_pred_sample;
%                     end
                    [y_pred_sample,~,~] = calc_ann(x(i,:),beetle_curr,activations);
                    err_curr = calc_mse(y_pred_sample,d);
                    if err_curr < prev_i_err 
                        i_best(:,j) = beetles(:,j);
                        prev_i_err(j) = err_curr;
                    end
                    if err_curr < g_err
                       g_best = beetles(:,j);
                       g_err = err_curr;
                       convergence_history((i-1)*n_beetles+j) = err_curr;
                    end
                end
                weights = pack_weights(g_best,weights_shape);
                
            elseif strcmp(opt,"BAS") == 1
                [weights] = BAS(weights,weights_shape, x(i,:),d(i,:),step,c1,c2,activations);
            else
                weights = update_weights(weights,x(i,:),d(i,:),learning_rate,activations);
            end
            [y,~,~] = calc_ann(x(i,:),weights,activations);
            y_pred(i,:) = y;

        end
        prev_error = error;
        error = calc_mse(y_pred,d);
        
%         fprintf("Epoch %d/%d  mse:%f\n",count,epochs,error)
        error_history = [error_history,error];
        if prev_error < error && strcmp(opt,"BSO") == 0 && strcmp(opt,"BAS") == 0
            weights = prev_weights;
            error = prev_error;
            error_history(end) = [];
            break
        end
        
        count = count + 1;

    end
%     if strcmp(opt,"BSO") == 1
%         error_history = convergence_history;
%     end
    
end
    