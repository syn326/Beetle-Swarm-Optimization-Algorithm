function [weights] = BAS(weights,weights_shape, X,y_true,step,c1,c2,activations)
    [y_pred,~,~] = calc_ann(X,weights,activations);    
    err_prev = calc_mse(y_pred,y_true);
    beetle = flatten_weights(weights);
    %(1)
    dir = rands(size(beetle,1),1)/norm(rands(size(beetle,1)),1);
    %(5)
    d = step / c2;
    %(2)
    v_xr = beetle + d * dir/2;
    v_xl = beetle - d * dir/2;
    
    %(3)
    xr = pack_weights(v_xr,weights_shape);
    xl = pack_weights(v_xl,weights_shape);
    %for plot 
%     figure(1)
%     ids = (1:size(beetle,1)/2);
%     plot(ids,beetle,'k',ids,v_xr,'b',ids,v_xl,'r') 
%     dim = floor(sqrt(size(beetle,1)));
%     z_beetle = reshape(beetle(1:dim*dim),[dim,dim]);
%     mesh(z_beetle)
%     pause(.1)

%     z_beetle = [mean(beetle(1:size(beetle,1)/2)),mean(beetle(size(beetle,1)/2+1:end))];
%     px = mean(beetle(1:size(beetle,1)/2))
%     py = mean(beetle(size(beetle,1)/2+1:end))
%     plot3(px,py,ids)
%     pause(.01)
%     mesh(beetle*beetle')
%     temp = floor(size(beetle,1)/3);
%     scatter3(mean(beetle((1:temp))),mean(beetle(temp:temp * 2)),mean(beetle((temp * 2:end))),'b')
%     hold on
%     gif
%     pause(.01)

    %end plot
    
    [y_pred_r,~,~] = calc_ann(X,xr,activations);
    [y_pred_l,~,~] = calc_ann(X,xl,activations);
    err_r = calc_mse(y_pred_r,y_true);
    err_l = calc_mse(y_pred_l,y_true);
    beetle_next = beetle - step * dir * sign(err_r - err_l);
    new_weights = pack_weights(beetle_next,weights_shape);
    [y_pred,~,~] = calc_ann(X,new_weights,activations);
    err_next = calc_mse(y_pred,y_true);
    if err_prev > err_next
        weights = new_weights;
    end
    
    %(4)

    
    
end
