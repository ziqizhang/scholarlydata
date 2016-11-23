package org.scholarlydata.feature.pair;

/**
 *
 */
public class SFSquareRoot implements SmoothingFunction {
    @Override
    public double apply(double d) {
        return Math.sqrt(d);
    }

    public String getName(){
        return "sqrt";
    }
}
