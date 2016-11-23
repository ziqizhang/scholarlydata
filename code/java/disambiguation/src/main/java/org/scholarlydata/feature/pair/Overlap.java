package org.scholarlydata.feature.pair;

import java.util.Collection;
import java.util.HashSet;
import java.util.Set;

/**
 *
 */
public class Overlap {

    private int option;
    private SmoothingFunction sf;

    /**
     * score is not smoothed.
     * @param option:
     *              0 - intersection/union
     *              1 - intersection/max{|obj1|, |obj2|}
     *              2 - 1 if intersection>0; 0 otherwise
     *
     */
    public Overlap(int option){
        this.option=option;
    }

    /**
     * score is smoothed
     * @param option
     * @param sf
     */
    public Overlap(int option, SmoothingFunction sf){
        this(option);
        this.sf=sf;
    }

    public int getOption(){
        return option;
    }

    public double score(Collection<String> obj1, Collection<String> obj2){
        if(obj1.size()==0 && obj2.size()==0)
            return 0;
        Set<String> inter = new HashSet<>(obj1);
        Set<String> union = new HashSet<>(obj1);
        inter.retainAll(obj2);
        union.addAll(obj2);

        double raw=0.0;

        switch (option){
            case 0:
                raw= (double)inter.size()/(union.size());
                break;
            case 1:
                raw= (double) inter.size()/Math.min(obj1.size(), obj2.size());
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
