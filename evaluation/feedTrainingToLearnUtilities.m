trainingData_Y = readcsv(~/Documents/Msc-project/) % n_tainingItems x 1
trainingData_X = readcsv(~/Documents/Msc-project/) % n_trainingItems x n_variables
% n_variables = 1 for coarse preferences

n_users = 10429;
user_id = readscv(encoded_user_id.csv)
items_of_user = cell(n_users)
for usr in 1:n_users:
    items_of_user{usr} = find(user_id==usr-1)
priorMEANs = cell(n_users);
priorVARs = cell(n_users);

domainCardinalities = n_coarseClasses;% I have assumed the same cardinality
% for each discrete variable domain. You can rewrite using cells instead of
% matrices to deal with problems where this is not the case. For the corase
% preferences case, there is only 1 variable, so this is not an issue.

% Set uninformed priors for each user:
for usr = 1:n_users
    priorMEANs{usr} = ones(n_variables, ...
        domainCardinalities) * (2.5 / n_variables);
    % Assuming ratings in [0, 5].
    priorVARs{usr} = ones(n_variables, ...
        domainCardinalities) * (1.25 / n_variables);
end % for each user


presetVariance = 0.2; % or whatever feels appropriate; something small.
usr = 1
getTrainingDataForUser % this depends on how your data is stored, but the
% idea is to only get the data points that correspond to the specific user,
% and only the ones in the training dataset.v
for itm = 1:items_of_user{usr}
    sumEvidence = struct( ...
        'adjustedMean', trainingData_Y(itm)/presetVariance, ...
        'precision', 1/presetVariance );
    
    [ priorMEANs{usr}, priorVARs{usr} ] = updateBelief( ...
        priorMEANs{usr}, priorVARs{usr}, ...
        trainingData_X(itm, :), sumEvidence);
    
end % for each rated item in the training set
