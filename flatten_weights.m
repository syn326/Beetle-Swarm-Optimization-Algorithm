function v_w = flatten_weights(A,ni,nh)
    v_w = [];
    for i = 1:size(A,2)
        temp =  reshape(A{i},[size(A{i},1)*size(A{i},2),1]);
        v_w = [v_w ; temp];
    end
end