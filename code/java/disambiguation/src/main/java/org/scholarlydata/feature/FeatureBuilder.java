package org.scholarlydata.feature;

import org.apache.commons.lang3.tuple.Pair;


/**
 * implementations of this class should create one type of features for each object
 */
public interface FeatureBuilder<FeatureType, T> {

    Pair<FeatureType, T> build(String objId);
}
