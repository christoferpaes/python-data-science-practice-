# -*- coding: utf-8 -*-
"""Untitled25.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1-s1kmQQmzXlm5lAWyP8WplYPRFu8wfx7
"""

from scipy.stats import chi2_contingency, ttest_ind

class YesPerson:
    def __init__(self, sex, levelofeducation, numberOfJobs, numberofCerts, countryOfOrigin):
        self.sex = sex
        self.levelofeducation = levelofeducation
        self.numberOfJobs = numberOfJobs
        self.numberofCerts = numberofCerts
        self.countryOfOrigin = countryOfOrigin

class NoPerson:
    def __init__(self, sex, levelofeducation, numberOfJobs, numberofCerts, countryOfOrigin):
        self.sex = sex
        self.levelofeducation = levelofeducation
        self.numberOfJobs = numberOfJobs
        self.numberofCerts = numberofCerts
        self.countryOfOrigin = countryOfOrigin

# Define instances for YesPerson
firstP = YesPerson('F', 'mastersofarts', 3, 18, 'UK')
secondP = YesPerson('M', 'IncompleteU', 4, 3, 'US')
thirdP = YesPerson('F', 'Incomplete', 5, 0, 'US')
fourthP = YesPerson('F', 'bachelorsofscience', 5, 0, 'US')
fifthP = YesPerson('M', 'bachelorsofscience', 0, 0, 'US')
sixthP = YesPerson('F', 'mastersofscience', 0, 0, 'US')
seventhP = YesPerson('M', 'incomplete', 4, 0, 'US')
eigthP = YesPerson('M', 'bachelorsofscience', 3, 0, 'US')
ninthP = YesPerson('M', 'phd', 3, 0, 'US')
tenthP = YesPerson('M','Incomplete', 1, 0, 'US')

# Define instances for NoPerson
nfirstP = NoPerson('M', 'bachelors', 7, 0, 'US')
nsecondP = NoPerson('M', 'bachelorofarts', 7, 0, 'US')
nthirdP = NoPerson('F', 'bachelorofarts', 2, 0, 'US')
nfourthP = NoPerson('M', 'bachelorofscience', 4, 4, 'Mexico')
nfifthP = NoPerson('M', 'mastersofscience', 6, 1, 'US')
nsixthP = NoPerson('M', 'mastersofscience', 1, 0, 'US')
nseventhP = NoPerson('M', 'associates', 5, 0, 'US')
neigthP = NoPerson('M', 'mastersofscience', 4, 0, 'US')
nninthP = NoPerson('M', 'bachelorsofarts', 5, 2, 'US')

# Define attributes
attributes = ['sex', 'levelofeducation', 'numberOfJobs', 'numberofCerts', 'countryOfOrigin']

def perform_chi_square(attribute):
    # Create a contingency table for the attribute
    table = [[sum(getattr(p, attribute) == val for p in [firstP, secondP, thirdP, fourthP, fifthP, sixthP, seventhP, eigthP, ninthP, tenthP]),
              sum(getattr(p, attribute) == val for p in [nfirstP, nsecondP, nthirdP, nfourthP, nfifthP, nsixthP, nseventhP, neigthP, nninthP])]
             for val in set(getattr(firstP, attribute) for firstP in [firstP, secondP, thirdP, fourthP, fifthP, sixthP, seventhP, eigthP, ninthP, tenthP])]

    # Perform Chi-squared test
    stat, p, dof, expected = chi2_contingency(table)
    return p

def perform_t_test(attribute):
    yes_vals = [getattr(p, attribute) for p in [firstP, secondP, thirdP, fourthP, fifthP, sixthP, seventhP, eigthP, ninthP, tenthP]]
    no_vals = [getattr(p, attribute) for p in [nfirstP, nsecondP, nthirdP, nfourthP, nfifthP, nsixthP, nseventhP, neigthP, nninthP]]

    stat, p = ttest_ind(yes_vals, no_vals)
    return p

# Test each attribute
for attribute in attributes:
    if isinstance(getattr(firstP, attribute), int):
        p_value = perform_t_test(attribute)
    else:
        p_value = perform_chi_square(attribute)

    print(f'Attribute: {attribute}, p-value: {p_value}')

from scipy.stats import chi2_contingency

# Define the data
data = [
    [4, 6],  # Yes vs No for sex
    [5, 5],  # Yes vs No for levelofeducation
    [5, 5],  # Yes vs No for numberOfJobs
    [5, 5],  # Yes vs No for numberofCerts
    [5, 5],  # Yes vs No for countryOfOrigin
]

# Perform chi-squared test
stat, p, dof, expected = chi2_contingency(data)

# Define significance level
alpha = 0.05

# Check if p-value is less than alpha
if p < alpha:
    print("There is a significant relationship between the attribute and the outcome.")
else:
    print("There is no significant relationship between the attribute and the outcome.")