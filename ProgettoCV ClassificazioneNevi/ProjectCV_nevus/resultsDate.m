function [accuracy, precision, recall]= resultsDate(classification, labels, stringTest)
%% Calculate accuracy and the matrix of confusion.

accuracy = mean(classification == labels);

cm = confusionchart(classification, labels);

cm.Title = (stringTest);
end
