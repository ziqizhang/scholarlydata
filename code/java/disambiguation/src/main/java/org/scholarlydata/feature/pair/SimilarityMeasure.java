package org.scholarlydata.feature.pair;

import java.util.Collection;

/**
 * Created by zqz on 01/04/17.
 */
public abstract class SimilarityMeasure {
    protected boolean normalizeScore;
    protected String option;
    protected SmoothingFunction sf;

    /**
     * score is not smoothed.
     *
     * @param option: 0 - intersection/union
     *                1 - intersection/min{|obj1|, |obj2|}
     *                2 - 1 if intersection>0; 0 otherwise
     */
    public SimilarityMeasure(String option, boolean normalizeScore) {
        this.option = option;
        this.normalizeScore = normalizeScore;
    }

    /**
     * score is smoothed
     *
     * @param option
     * @param sf
     */
    public SimilarityMeasure(String option, boolean applyEquatreScoreNorm, SmoothingFunction sf) {
        this(option, applyEquatreScoreNorm);
        this.sf = sf;
    }

    public String getOption() {
        return option;
    }

    public SmoothingFunction getSf() {
        return sf;
    }

    public abstract double score(Collection<String> obj1, Collection<String> obj2);
}