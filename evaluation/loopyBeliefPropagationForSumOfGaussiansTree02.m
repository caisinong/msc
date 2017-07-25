function [ posteriorNodePotentials, IS_1_belief, ...
    messagesFactorToVariable, IS_1_messagesFactorToVariable, ...
    messagesVariableToFactor, IS_1_messagesVariableToFactor] = ...
    loopyBeliefPropagationForSumOfGaussiansTree02( priorNodePotentials, ...
    sumEvidence, numberOfIterations)
%LOOPYBELIEFPROPAGATION Summary of this function goes here
%   priorNodePotentials: array of structs, of length n_variables, with 
%                        the prior belief over each node, as given by the
%                        fields .adjustedMean, .precision. The last node
%                        always refers to the sum of Gaussians.
%
%   All factors and messages are Gaussians, unless they are $1$s. The
%   latter is indicated in a separate matrix for each belief/message
%   structure (i.e. beliefs, messages from factors to variables, and 
%   messages from variables to factors).
    
    n_variables = length(priorNodePotentials) + 1;
    %n_features  = n_variables - 1;
    n_factors   = n_variables + 1;
    
    %numberOfIterations = 1; T = 1
    
    factorGraph = eye(n_variables, n_factors);
    factorGraph(:, n_factors - 1) = 1;
    factorGraph(n_variables, n_factors) = 1;
    
    %% Initialise Messages & Belief:
    messagesFactorToVariable = struct('precision', 0, 'adjustedMean', 1); % I don't need to maintain old/new vectors?
    IS_1_messagesFactorToVariable = ones(n_factors, n_variables); % To indicate messages of $1$.
    messagesVariableToFactor = struct('precision', 0, 'adjustedMean', 1);
    IS_1_messagesVariableToFactor = ones(n_variables, n_factors); % To indicate messages of $1$.

    belief = struct('precision', 0, 'adjustedMean', 1);
    IS_1_belief = ones(1, n_variables); % To indicate belief of $1$. Used in initial loop...
    
     for variable = 1:n_variables
        
        belief(variable) = struct('precision', 0, 'adjustedMean', 0);
        
        for factor = 1:n_factors
            
            if factorGraph(variable, factor)
                messagesFactorToVariable(factor, variable) = ...
                    struct('precision', -1, 'adjustedMean', 1);
                messagesVariableToFactor(variable, factor) = ...
                    struct('precision', -1, 'adjustedMean', 1);
            else
                messagesFactorToVariable(factor, variable) = ...
                    struct('precision', -1, 'adjustedMean', -1); % unnecessarily sparse
                messagesVariableToFactor(variable, factor) = ...
                    struct('precision', -1, 'adjustedMean', -1);
                
            end
        end
     end
    
    % Incorporate evidence over sum variable into the belief/potential?
    % - No. Factor added for the evidence.
%     belief(n_variables) = ...  This is wrong!
%         struct( 'precision', 10, 'adjustedMean', sumEvidence );
%     IS_1_belief(n_variables) = 0;
    
    %% Run Loopy Belief Propagation:
    for iteration = 1:numberOfIterations updated information into the posterior:
	
        %% Simplified Message Passing:
        % 1st - From Partial Assignment Priors to variables 
        %       ( $m_{ f_{ix_i} \rightarrow u_{ix_i} }$ ):
        
        % 2nd - From the Sum Factor to the outcome utility
        %       ( $m_{ f_{\sum} \rightarrow u }$ ):
        
        % 3rd - From the Sum Factor to the partial outcome utilities:
        %       ( $m_{ f_{\sum} \rightarrow u_i }, \forall i$ ):
        % I can approximate u_{\response/true} with a Gaussian centered on
        % the response and with a small/negligible variance.
        
        %% Compute messages from variables to factors: This should be interleaved I believe...
        % (All messages are 1s at start, so nothing happens here at the 
        % first iteration.)
        for variable = 1:n_variables
            for factor = 1:n_factors
                
                if factorGraph(variable, factor) % this edge exists
                
                    precision = 0;
                    adjustedMean = 0;
                    
                    flag_isNoLonger_1 = 0;
                    
                    for otherFactor = 1:n_factors
                        if otherFactor ~= factor && ...
                                factorGraph(variable, otherFactor)
                            
                            if ~IS_1_messagesFactorToVariable(...
                                    otherFactor, variable)
                                
                                flag_isNoLonger_1 = 1; % even if it wasn't
                                
                                precision = precision + ...
                                    messagesFactorToVariable(...
                                    otherFactor, variable).precision;
                                
                                adjustedMean = adjustedMean + ...
                                    messagesFactorToVariable(...
                                    otherFactor, variable).precision * ...
                                    messagesFactorToVariable(...
                                    otherFactor, variable).adjustedMean;
                                
                            end % if this isn't a message of $1$
                            
                        end % if this is not the same factor
                        % && the edge exists
                    end % for each other factor
                    
                    if flag_isNoLonger_1
                        
                        IS_1_messagesVariableToFactor(...
                            variable, factor) = 0;
                        
                        adjustedMean = adjustedMean / precision;
                        
                        messagesVariableToFactor(variable, factor...
                            ).precision = precision;
                        messagesVariableToFactor(variable, factor...
                            ).adjustedMean = adjustedMean;
                        
                    end % if we received a non $1$ message
                    
                end % if this edge exists
                
%                 variable, factor
%                 messagesVariableToFactor(variable, factor)
                
            end % for each factor
        end % for each variable
        
        % At this point, some messages (all in the first iteration) will
        % still be 1s, rather than Normal distributions.
        
        %% Compute messages from factors to variables:
        % Root factors to features & Sum factor to sum:
        for factor = 1:n_factors
            for variable = 1:n_variables
            
                if factorGraph(variable, factor) % this edge exists
                    if factor < n_factors - 1 % for all root factor to
                        % feature vars
                        
                        [ messagesFactorToVariable(factor, ...
                            variable).precision, ...
                            messagesFactorToVariable(factor, ...
                            variable).adjustedMean, ...
                            IS_1_messagesFactorToVariable(...
                            factor, variable) ] = doRootUpdate( ...
                            priorNodePotentials(variable).precision, ...
                            priorNodePotentials(variable).adjustedMean );
                        
                        % Immediately update corresponding beliefs here?
                        % - Yes!
                        [ belief, IS_1_belief ] = updateBelief( ...
                            variable, factorGraph, ...
                            messagesFactorToVariable, ...
                            IS_1_messagesFactorToVariable, ...
                            belief, IS_1_belief );
                        
%                           166, factor, variable
%                           messagesFactorToVariable(factor, variable)
%                         
%                         belief(variable)
%                         
%                         IS_1_belief(variable)
                        
                    elseif variable < n_variables % if this connects sum
                        % factor to feature
                        
                        % These are to be done !after! the sum factor to
                        % sum variable has been computed:
                        
                        %                     [ multiplicativeFactor ] = ...
                        %                         transformSum( n_variables, variable );
                        %
                        %                     [ messagesFactorToVariable(factor, ...
                        %                         variable).precision, ...
                        %                         messagesFactorToVariable(factor, ...
                        %                         variable).adjustedMean, ...
                        %                         IS_1_messagesFactorToVariable(...
                        %                         factor, variable) ] = doSumUpdate( belief, ...
                        %                         IS_1_belief, messagesFactorToVariable, ...
                        %                         IS_1_messagesFactorToVariable, ...
                        %                         multiplicativeFactor, factor, variable );
                        
                    elseif factor < n_factors % if this connects sum factor
                        % to sum variable, i.e. (factor == n_factors)
                        % && (variable == n_variables)
                        
%                         195, belief(1), belief(2), belief(3), belief(4),
%                         IS_1_belief, 
                        
                        multiplicativeFactor = ones(1, n_variables);
                        multiplicativeFactor(n_variables) = 0;
                        
%                         multiplicativeFactor
%                         
%                         IS_1_messagesFactorToVariable
                        
                        [ messagesFactorToVariable(factor, ...
                            variable).precision, ...
                            messagesFactorToVariable(factor, ...
                            variable).adjustedMean, ...
                            IS_1_messagesFactorToVariable(...
                            factor, variable) ] = doSumUpdate( belief, ...
                            IS_1_belief, messagesFactorToVariable, ...
                            IS_1_messagesFactorToVariable, ...
                            multiplicativeFactor, factor, variable );
 
%                         209, factor, variable
%                         messagesFactorToVariable(factor, variable)
%                        
%                         messagesFactorToVariable(factor, ...
%                             variable)
%                         
%                         IS_1_messagesFactorToVariable(...
%                             factor, variable)
                        
                        % Update belief for sum variable:
                        [ belief, IS_1_belief ] = updateBelief( ...
                            variable, factorGraph, ...
                            messagesFactorToVariable, ...
                            IS_1_messagesFactorToVariable, ...
                            belief, IS_1_belief  );
                        
%                         belief(variable)
%                         
%                         IS_1_belief(variable)
                        
                    else % if this connects evidence factor to sum variable
                        
                        [ messagesFactorToVariable(factor, ...
                            variable).precision, ...
                            messagesFactorToVariable(factor, ...
                            variable).adjustedMean, ...
                            IS_1_messagesFactorToVariable(...
                            factor, variable) ] = doRootUpdate( ...
                            sumEvidence.precision, ...
                            sumEvidence.adjustedMean );
                        
                        % Immediately update corresponding beliefs here?
                        % - Yes!
                        [ belief, IS_1_belief ] = updateBelief( ...
                            variable, factorGraph, ...
                            messagesFactorToVariable, ...
                            IS_1_messagesFactorToVariable, ...
                            belief, IS_1_belief );
                        
%                         248, factor, variable
%                         messagesFactorToVariable(factor, variable)
%                         
%                         belief(variable)
%                         
%                         IS_1_belief(variable)
                        
                    end % if this is a feature variable
                end % if this edge exists  
                
            end % for each variable
        end % for each factor
        
        % Sum factor to features:
        factor = n_factors - 1;
        for variable = 1:n_variables-1
            if factorGraph(variable, factor) % this edge exists
                
                [ multiplicativeFactor ] = ...
                    transformSum( n_variables, variable );
                
                [ messagesFactorToVariable(factor, ...
                    variable).precision, ...
                    messagesFactorToVariable(factor, ...
                    variable).adjustedMean, ...
                    IS_1_messagesFactorToVariable(...
                    factor, variable) ] = doSumUpdate( belief, ...
                    IS_1_belief, messagesFactorToVariable, ...
                    IS_1_messagesFactorToVariable, ... 
                    multiplicativeFactor, factor, variable );
                
                % Update belief for variable:
                [ belief, IS_1_belief ] = updateBelief( ...
                    variable, factorGraph, ...
                    messagesFactorToVariable, ...
                    IS_1_messagesFactorToVariable, ... not working??
                    belief, IS_1_belief  );
                
%                 286, factor, variable
%                 messagesFactorToVariable(factor, variable)
%                
%                 messagesFactorToVariable(factor, ...
%                     variable)
%                 
%                 IS_1_messagesFactorToVariable(...
%                     factor, variable)
%                 
%                 belief(variable)
%                 
%                 IS_1_belief(variable)
                
            end % if that connects to the sum factor
        end % for each variable except the sum variable
        
        % Now, we only need to update beliefs that utilise these newly
        % computed messages. Basically, update beliefs as new messages from
        % factor to variable are computed.
        
    end % for each iteration / could change for convergence but might not need to
    
    %% Normalise beliefs (which are Gaussian distributions):
    % for each node/variable:
    
        % create Normal distribution struct:
    
        % feed it to a Gaussian integral calculator:
    
        % divide the gaussians with this
    
     % - - - % Alternatively, we could just run a normalisation funtion over it...
    
    %% Prepare output:
    posteriorNodePotentials = belief;
    %IS_1_belief
    
end

function [ belief, IS_1_belief ] = updateBelief( variable, factorGraph, ...
    messagesFactorToVariable, IS_1_messagesFactorToVariable, ... Something's wrong...
    belief, IS_1_belief  )
           
    [ ~, n_factors ] = size(factorGraph);

    variance = 0;
    mean = 0;
    
    flag_isNoLonger_1 = 0;
    isFirstMessage = 1;
%     variable
    for factor = 1:n_factors
%         factor
%         factorGraph(variable, factor)
        if factorGraph(variable, factor)
%            IS_1_messagesFactorToVariable(factor, variable)
            if ~IS_1_messagesFactorToVariable(factor, variable)
                
                flag_isNoLonger_1 = 1; % even if it wasn't
                
                oldMean = mean;
                oldVariance = variance;
                
%                 messagesFactorToVariable(...
%                     factor, variable).adjustedMean
%                 messagesFactorToVariable(factor, ...
%                     variable).precision
                
                newMean = messagesFactorToVariable(...
                    factor, variable).adjustedMean / ...
                    messagesFactorToVariable(factor, ...
                    variable).precision;
                
                newVariance = 1 /...
                    messagesFactorToVariable(...
                    factor, variable).precision;
%                 isFirstMessage
                if isFirstMessage
                
                    mean = newMean;
                    variance = newVariance;
                    
                    isFirstMessage = 0;
                    
                else                  
%                     oldMean, newMean, oldVariance, newVariance
                    mean = ( oldMean * newVariance + ...
                        newMean * oldVariance ) / ...
                        ( newVariance + oldVariance );
                    
                    variance = ( oldVariance * newVariance ) / ...
                        ( oldVariance + newVariance );
                
                    
%                 adjustedMean = adjustedMean + ...
%                     messagesFactorToVariable(... adjMean = precision*mean
%                     factor, variable).precision * ...
%                     messagesFactorToVariable(...
%                     factor, variable).adjustedMean;
                end
                
            end % if the message from this factor to the variable
            % is not a $1$ (as oppossed to a Gaussian)
        end % if this factor is in the neighborhood of the variable
    end % for each factor
%     flag_isNoLonger_1
    if flag_isNoLonger_1
        
        IS_1_belief(variable) = 0;
%                 variable
%                 variance, mean
        precision = 1 / variance;
        adjustedMean = mean * precision;
%       377, variable
        belief(variable).precision    = precision;
        belief(variable).adjustedMean = adjustedMean;
%         belief(variable)
        
    end % if we received a non $1$ message
        
    % Could normalise here, but I will keep it for the end, since there
    % is no need of it yet.
    
    % During the first iteration, these beliefs would all still be 1s,
    % rather than Normal distributions. (Shouldn't be true anymore...)
                
end

function [ precision, adjustedMean, IS_1 ] = doRootUpdate( ...
    factorPrecision, factorAdjustedMean )
    
    IS_1 = 0;
    
%     factorVariance = 1 / factorPrecision;
%     factorMean     = factorAdjustedMean * factorPrecision;
%     
%     precision    =          1 / factorVariance;
%     adjustedMean = factorMean / factorVariance;
    
    precision    = factorPrecision;
    adjustedMean = factorAdjustedMean;
    
end % function doRootUpdate

function [ precision, adjustedMean, IS_1 ] = ... I think only the following two functions will actually be needed...
    doSumUpdate( belief, IS_1_belief, messagesFactorToVariable, ...
    IS_1_messagesFactorToVariable, multiplicativeFactor, factor, variable ) % belief gives me the precision and adjustedMean at each node     

    n_variables = length(multiplicativeFactor);
    
    IS_1 = 1;
    
    precision = 0;
    adjustedMean = 0;
    
%     disp(' - Now running new call to doSumUpdate - ')
%     420, variable
%     factor
%     disp(' - - - - - - - - - - - - - - - - - - - - ')
    for otherVariable = 1:n_variables  
        if variable ~= otherVariable
            % All the following if neither beliefs, nor any message are 
            % 1s. Problems will arise if these are not 1s at the same 
            % time...
%             otherVariable
%             variable
%             IS_1_belief(otherVariable)
%             IS_1_messagesFactorToVariable(factor, otherVariable)
            
            if ~IS_1_belief(otherVariable)
                beliefOtherPrecision = belief(otherVariable).precision;
                beliefOtherAdjedMean = belief(otherVariable).adjustedMean;
            else % if the belief is a constant 1 (typically by 
                % instantiation)
                beliefOtherPrecision = 0;
                beliefOtherAdjedMean = 0;
            end % if the belief is Gaussian
            
            if ~IS_1_messagesFactorToVariable(factor, otherVariable)
                messageOtherPrecision = messagesFactorToVariable(...
                    factor, otherVariable).precision;
                messageOtherAdjedMean = messagesFactorToVariable(...
                    factor, otherVariable).adjustedMean;
            else % if the message is a constant 1 (typically by 
                % instantiation)
                messageOtherPrecision = 0;
                messageOtherAdjedMean = 0;
            end % if the message is Gaussian
            
            if IS_1_belief(otherVariable) && ...
                    IS_1_messagesFactorToVariable(factor, otherVariable)
                % do nothing
            else % if the computation can go ahead
                IS_1 = 0;

%                 otherVariable
%                 
%                 beliefOtherPrecision
%                 beliefOtherAdjedMean
%                 
%                 messageOtherPrecision
%                 messageOtherAdjedMean
%                 
%                 multiplicativeFactor(otherVariable)
                
                precision = precision + ...
                    multiplicativeFactor(otherVariable)^2 * ...
                    (1 / (beliefOtherPrecision - messageOtherPrecision) );
                
                adjustedMean = adjustedMean + ...
                    multiplicativeFactor(otherVariable) ... (beliefOtherAdjedMean - messageOtherAdjedMean) = 0 for the sum factor -> sum variable
                    * ...
                    (beliefOtherAdjedMean - messageOtherAdjedMean) ...
                    * ...
                    ( 1 / (beliefOtherPrecision - messageOtherPrecision) );
            end % if both node and message potential is non Gaussian 
            % (constant 1) 
        end % if that is another variable
    end % for each variable
    % variable; factor == 4, 4 - > precision = 0
    precision = 1 / precision;
    
    adjustedMean = precision * adjustedMean;
    
end % function doSumUpdate

function [ factor ] = transformSum( n_variables, variableIndex )
    
    factor                = -1 * ones(1, n_variables);
    factor(variableIndex) = 0;
    factor(n_variables)   = 1;
    
end % function transformSum
