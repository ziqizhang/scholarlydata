package org.scholarlydata.feature.pair;

import org.apache.commons.lang3.tuple.Pair;
import org.scholarlydata.feature.FeatureType;

import java.util.Map;

/**
 *
 */
public interface PairFeatureBuilder {

    Map<Pair<FeatureType, String>, Double> build(String obj1, String obj2);
}
