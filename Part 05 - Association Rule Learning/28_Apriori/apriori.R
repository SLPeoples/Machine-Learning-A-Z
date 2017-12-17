# Apriori

#Optimizing placement of products in a store based on items bought together.

# Data Preprocessing
# install.packages('arules')
library(arules)
dataset = read.csv('Market_Basket_Optimisation.csv', header = FALSE)
dataset = read.transactions('Market_Basket_Optimisation.csv', sep = ',', rm.duplicates = TRUE)
summary(dataset)
itemFrequencyPlot(dataset, topN = 10)

# Training Apriori on the dataset
# Support: left of position on itemFrequencyPlot choose cutoff
#   Items purchased four times per day
#     (4*7) = 28 / Total num Transactions 
#     28 / 7500 = .0037 ~ .004; 
# Confidence: work the confidence down from default .8
#   Check rules. .8: 0 rules
#                .4: 281 rules ; Chocolate was overprepresented.
#                .2: 811 rules
rules = apriori(data = dataset, parameter = list(support = 0.004, confidence = 0.2))

# Visualising the results
inspect(sort(rules, by = 'lift')[1:10])
