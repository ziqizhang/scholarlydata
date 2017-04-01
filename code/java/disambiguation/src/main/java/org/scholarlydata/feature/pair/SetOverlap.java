package org.scholarlydata.feature.pair;

import org.apache.commons.collections.CollectionUtils;

import java.util.Collection;
import java.util.HashSet;
import java.util.Set;

/**
 *
 */
class SetOverlap extends SimilarityMeasure {


    /**
     * score is not smoothed.
     *
     * @param option: 0 - intersection/union
     *                1 - intersection/min{|obj1|, |obj2|}
     *                2 - 1 if intersection>0; 0 otherwise
     */
    public SetOverlap(String option, boolean normalizeScore) {
        super(option, normalizeScore);

    }

    /**
     * score is smoothed
     *
     * @param option
     * @param sf
     */
    public SetOverlap(String option, boolean applyEquatreScoreNorm, SmoothingFunction sf) {
        this(option, applyEquatreScoreNorm);
        this.sf = sf;
    }

    public String getOption() {
        return option;
    }

    public SmoothingFunction getSf() {
        return sf;
    }

    public double score(Collection<String> obj1, Collection<String> obj2) {
        if (obj1.size() == 0 && obj2.size() == 0)
            return 0;
        Set<String> obj1S = new HashSet<>(obj1);
        Set<String> obj2S = new HashSet<>(obj2);
        Collection<String> union = CollectionUtils.union(obj1S, obj2S);
        Collection<String> inter = CollectionUtils.intersection(obj1S, obj2S);
        if (union.size() == 0 || inter.size() == 0)
            return 0.0;

        double raw = 0.0;

        if (option.equals("union"))
            raw = (double) inter.size() / (union.size());
        else if (option.equals("max")) {
            int min = Math.min(obj1.size(), obj2.size());
            raw = (double) inter.size() / min;
            if (normalizeScore)
                raw = EquatreNormalizerFactor.applyLogisticNorm(min, raw);
        } else if (option.equals("hasInter"))
            raw = inter.size() > 0 ? 1.0 : 0.0;

        if (sf == null)
            return raw;
        return sf.apply(raw);
    }

}
