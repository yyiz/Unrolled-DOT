function loadStruct(s)
    vars = fieldnames(s);
    for i = 1:length(vars)
        assignin('caller', vars{i}, s.(vars{i}));
    end
end