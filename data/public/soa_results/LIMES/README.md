You need the following to re-produce LIMES experiments:

1. point your config files to config_org.xml for ORG, or config_per.xml for PER.
2. correct relevant file paths in the config files to point to the source/target, and training files (trainAndTestSplits)
3. use the correct ML algorithms (wombat simple, wombat complete).

for 1-3, please read LIMES manual for details.

Next, place 'SparqlQueryModule.java' into the package folder of your LIMES distribution to "org.aksw.limes.core.io.query", to replace the original one. This file has a workaround to address a slow graph flattening issue. Build your LIMES and run experiments uisng the configs.

To evlauate, check out https://github.com/ziqizhang/scholarlydata, use the class org.scholarlydata.exp.limes.Evaluator.

In fact, this folder also includes output produced by LIMES, under 'output'. We ran Wombat simple/complete with a threshold of 0.1. The evaluator will then compute PRF applying different thresholds to this output.


NOTE: LIMES only predict positive matching pairs (predictions). To evaluate LIMES against negative pairs, given a test dataset (ground truth), any pairs in the ground truth but are not in predictions are considered as negative pairs.


