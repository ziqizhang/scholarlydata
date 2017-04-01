package org.scholarlydata.feature.pair;

import org.simmetrics.StringMetric;
import org.simmetrics.metrics.Levenshtein;

import java.util.Collection;

/**
 *
 */
public class StringSim extends SimilarityMeasure {
    private StringMetric metric;

    public StringSim(String option, boolean normalizeScore) {
        super(option, normalizeScore);
        if(option.equals("lev"))
            metric=new Levenshtein();
    }

    public StringSim(String option, boolean normalizeScore, SmoothingFunction sf) {
        this(option, normalizeScore);
        this.sf=sf;
    }

    @Override
    public double score(Collection<String> obj1, Collection<String> obj2) {
        Collection<String> smallSet = obj1.size()>obj2.size()?obj2:obj1;
        Collection<String> largeSet = obj1.size()>obj2.size()?obj1:obj2;

        double final_sum=0.0;
        for(String s1: smallSet){
            double total=0.0;
            for(String s2: largeSet){
                double score = metric.compare(s1, s2);
                total+=score;
            }
            double avg = total/largeSet.size();
            final_sum+=avg;
        }
        return smallSet.size()==0?0: final_sum/smallSet.size();
    }
}
