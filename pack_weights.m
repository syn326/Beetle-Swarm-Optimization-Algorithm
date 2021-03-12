function packed_weights = pack_weights(v_w,shpae_weights)
    packed_weights = {};
    left = 0;
    right = 0;
    for i = 1:size(shpae_weights,2)
        shape_curr = shpae_weights{i};
        left = right + 1;
        right = right + shape_curr(1) * shape_curr(2);
        packed_weights{end+1} = reshape(v_w(left:right),[shape_curr]);
end