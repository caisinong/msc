function [ posteriorMEANs, posteriorVARs  ] = updateBelief( priorMEANs, priorVARs, x, sumEvidence )
%UPDATEBELIEF
% sumEvidence = struct( 'adjustedMean', mean*precision, 'precision', 1/variance );

	%posteriorMEANs = priorMEANs;
	%posteriorVARs = priorVARs;
	
	% Create 'priorNodePotentials' by selecting from 'priorMEANs' and 'priorVARs' according to 'x':
	priorNodePotentials = selectPotentials( priorMEANs, priorVARs, x );
	
	% Update the respective part of the prior:
	[ posteriorNodePotentials, ~, ~, ~, ~, ~] = ...
	                loopyBeliefPropagationForSumOfGaussiansTree02( priorNodePotentials, sumEvidence, 1 );

	% Copy the updated information into the posterior:
	[ posteriorMEANs, posteriorVARs ] = ...
		copyToPotentials( posteriorNodePotentials, priorMEANs, priorVARs, x );

end % updateBelief

function priorNodePotentials = selectPotentials( priorMEANs, priorVARs, x )
%SELECTPOTENTIALS
	
	priorNodePotentials = struct('precision', 1, 'adjustedMean', 1);
	n_variables = length( priorMEANs(:, 1) );

	for variable = 1:n_variables
		priorPrecision = 1 / priorVARs( variable, x(variable) );
		priorAdjustedMean = priorPrecision  * priorMEANs( variable, x(variable) );
    		priorNodePotentials(variable) = ...
			struct( 'precision', priorPrecision, ...
        		'adjustedMean', priorAdjustedMean );
	end % for each variable

end % selectPotentials

function [ posteriorMEANs, posteriorVARs ] = copyToPotentials( posteriorNodePotentials, priorMEANs, priorVARs, x )
%COPYTOPOTENTIALS

	posteriorMEANs = priorMEANs;
	posteriorVARs = priorVARs;

    n_variables = length( priorMEANs(:, 1) );
    
	for variable = 1:n_variables
		VAR = 1 / posteriorNodePotentials(variable).precision;
		MEAN = posteriorNodePotentials(variable).adjustedMean * VAR;
		posteriorMEANs( variable, x(variable) ) = MEAN;
		posteriorVARs( variable, x(variable) ) = VAR;
	end % for each variable
	
end % copyToPotentials