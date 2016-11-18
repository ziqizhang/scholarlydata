package org.scholarlydata.feature.per;

import org.apache.commons.lang3.tuple.ImmutablePair;
import org.apache.commons.lang3.tuple.Pair;
import org.apache.jena.query.ResultSet;
import org.scholarlydata.SPARQLQueries;
import org.scholarlydata.feature.FeatureBuilderSPARQL;
import org.scholarlydata.feature.FeatureType;
import org.scholarlydata.feature.Predicate;

import java.util.List;

/**
 *
 */
public class FBPerPublishedWorkURI extends FeatureBuilderSPARQL<FeatureType, List<String>> {
    @Override
    public Pair<FeatureType, List<String>> build(String objId) {
        String queryStr = SPARQLQueries.getObjectsOf(objId,
                Predicate.PERSON_made.getURI());

        ResultSet rs = query(queryStr);
        return new ImmutablePair<>(FeatureType.PERSON_PUBLICATION_URI, getListResult(rs));
    }
}
