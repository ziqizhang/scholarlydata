#Silk experiment
This folder contains data to reproduce experiments on the [SILK](http://silkframework.org/) framework.

The file [scholarlySILK.zip](scholarlySILK.zip) contains the silk workspace. 
This has been exported directly from SILK, therefore it can be directly imported in the framework.
It contains the rules to match PERSONS and ORGANIZATION, as well as the source and target datasets formatted in SILK specific formats.

The folder [trainAndTestSplits](trainAndTestSplits) contains the train/test files that we used, both for [PERSON](trainAndTestSplits/PER) and [ORGANIZATION](trainAndTestSplits/ORG).

The folder [SILK-output](SILK-output) contains the results produced by SILK on all the testing splits, both for [PERSON](SILK-output/PER) and [ORGANIZATION](SILK-output/ORG).

The folder [SILK-eval-results](SILK-eval-results) contains the summaries of evaluation, both for [PERSON](SILK-eval-results/PER) and [ORGANIZATION](SILK-eval-results/ORG).
