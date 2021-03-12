function [beetles,V] = BSO(X,y_true,beetles,V,weights_shape,step,n_beetles,c1,c2,V_max,V_min,lambda,epoch,max_epochs,i_best, g_best, activations)
    omega_max = 0.9;
    omega_min = 0.4;
    figure(1)
    color = ['r','g','b','k','y'];
    temp = floor(size(beetles,1)/3);
        %(8)
        omega = omega_max - (omega_max-omega_min)*epoch/max_epochs;
        %(5)
        dist = step / c2;
        prev_err = Inf(1,n_beetles);
        for i = 1:n_beetles
            v_beetle_curr = beetles(:,i);
%             scatter3(mean(v_beetle_curr((1:temp))),mean(v_beetle_curr(temp:temp * 2)),mean(v_beetle_curr((temp * 2:end))),color(i))
%             gif
%             hold on
            
%             disp(size(beetles))
%             disp(weights_shape)
%             disp(size(v_beetle_curr))
            beetle_curr = pack_weights(v_beetle_curr,weights_shape);
            [y_pred,~,~] = calc_ann(X,beetle_curr,activations);
            prev_err(i) = calc_mse(y_pred,y_true);
            %(10)get antennae position x_r , x_l
            v_xr = v_beetle_curr + V(:,i) * dist / 2;
            v_xl = v_beetle_curr - V(:,i) * dist / 2;
            xr = pack_weights(v_xr,weights_shape);
            xl = pack_weights(v_xl,weights_shape);
            %(9) Update the incremental function
            [y_pred_r,~,~] = calc_ann(X,xr,activations);
            [y_pred_l,~,~] = calc_ann(X,xl,activations);
            err_r = calc_mse(y_pred_r,y_pred);
            err_l = calc_mse(y_pred_l,y_pred);
            zeta = step * V(:,i) * sign(err_r-err_l);
            
            %(7)
            r1 = rand(1);
            r2 = rand(1);
            V(:,i) = omega*V(:,i)+c1*r1*(i_best(:,i)-beetles(:,i))+c2*r2*(g_best-beetles(:,i));
            id = find(abs(V) < V_min);
            V(id) = sign(V(id)) * V_min;
            id = find(abs(V) > V_max);
            V(id) = sign(V(id)) * V_max;
            %(6) Update the position of the current search agent
            beetles(:,i) = v_beetle_curr + lambda * V(:,i) + (1-lambda) * zeta;
        end
        
 
    new_weights = pack_weights(g_best,weights_shape);
end
