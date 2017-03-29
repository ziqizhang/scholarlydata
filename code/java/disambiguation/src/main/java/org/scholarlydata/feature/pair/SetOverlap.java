package org.scholarlydata.feature.pair;

import org.apache.commons.collections.CollectionUtils;

import java.util.Collection;
import java.util.HashSet;
import java.util.Set;

/**
 *
 */
public class SetOverlap {

    protected int option;
    protected SmoothingFunction sf;

    /**
     * score is not smoothed.
     * @param option:
     *              0 - intersection/union
     *              1 - intersection/min{|obj1|, |obj2|}
     *              2 - 1 if intersection>0; 0 otherwise
     *
     */
    public SetOverlap(int option){
        this.option=option;
    }

    /**
     * score is smoothed
     * @param option
     * @param sf
     */
    public SetOverlap(int option, SmoothingFunction sf){
        this(option);
        this.sf=sf;
    }

    public int getOption(){
        return option;
    }
    public SmoothingFunction getSf(){
        return sf;
    }

    public double score(Collection<String> obj1, Collection<String> obj2){
        if(obj1.size()==0 && obj2.size()==0)
            return 0;
        Set<String> obj1S = new HashSet<>(obj1);
        Set<String> obj2S = new HashSet<>(obj2);
        Collection<String> union= CollectionUtils.union(obj1S, obj2S);
        Collection<String> inter=CollectionUtils.intersection(obj1S, obj2S);
        if(union.size()==0||inter.size()==0)
            return 0.0;

        double raw=0.0;

        switch (option){
            case 0:
                raw= (double)inter.size()/(union.size());
                break;
            case 1:
                int min  = Math.min(obj1.size(), obj2.size());
                raw= (double) inter.size()/min;
                raw = EquatreNormalizerFactor.applyLogisticNorm(min, raw);
                break;
            case 2:
                raw= inter.size()>0?1.0:0.0;
                break;
        }

        if(sf==null)
            return raw;
        return sf.apply(raw);
    }

}
