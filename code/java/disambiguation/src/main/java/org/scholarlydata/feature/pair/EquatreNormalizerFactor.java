package org.scholarlydata.feature.pair;

/**
 */
public class EquatreNormalizerFactor {

    protected static final double applyLogisticNorm(int evidenceCount,
                                                    double scoreToNormalize){
        double learningRate_threshold=10;
        double exp_r1 = Math.exp(0 - (evidenceCount - learningRate_threshold/2.0));
        double conf = 1.0 / (1 + exp_r1);
        return scoreToNormalize*conf;
    }
}
