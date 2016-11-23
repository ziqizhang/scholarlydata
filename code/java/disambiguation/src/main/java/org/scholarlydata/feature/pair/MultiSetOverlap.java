package org.scholarlydata.feature.pair;

import java.util.*;

/**
 * Created by zqz on 23/11/16.
 */
public class MultiSetOverlap extends SetOverlap {
    public MultiSetOverlap(int option) {
        super(option);
    }

    public MultiSetOverlap(int option, SmoothingFunction sf) {
        super(option, sf);
    }

    public double score(Collection<String> obj1, Collection<String> obj2){
        if(obj1.size()==0 && obj2.size()==0)
            return 0;
        List<String> inter = new ArrayList<>(obj1);
        List<String> union = new ArrayList<>(obj1);
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
