package org.scholarlydata.feature.pair;

import org.apache.commons.collections.CollectionUtils;

import java.util.*;

/**
 * Created by zqz on 23/11/16.
 */
public class MultiSetOverlap extends SetOverlap {
    public MultiSetOverlap(String option, boolean applyEquatreScoreNorm) {
        super(option, applyEquatreScoreNorm);
    }

    public MultiSetOverlap(String option, boolean applyEquatreScoreNorm,
                           SmoothingFunction sf) {
        super(option, applyEquatreScoreNorm, sf);
    }

    public double score(Collection<String> obj1, Collection<String> obj2) {
        if (obj1.size() == 0 && obj2.size() == 0)
            return 0;
        Collection<String> inter = CollectionUtils.intersection(obj1, obj2);
        Collection<String> union = CollectionUtils.union(obj1, obj2);

        if (inter.size() == 0 || union.size() == 0)
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
