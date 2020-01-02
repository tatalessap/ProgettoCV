function [classificationVoting] = voting(classification, class)
%{    
considering each classification, check how they classified the images and obtain a new classification through voting
%}

[M, N] = size(classification);

classificationVoting = classification(:,3); %initialization

votigA=0;
votigB=0;

%% Voting 
for i=1:M
    for j=1:N
        if(classification(i,j)==char(class(1)))
            votigA=votigA+1;
        else
            votigB=votigB+1;
        end
    end

    if (votigA>votigB)
        classificationVoting(i) = class(1);
    else
        classificationVoting(i) = class(2);
    end
    
    votigA=0;
    votigB=0;
    
end

end

