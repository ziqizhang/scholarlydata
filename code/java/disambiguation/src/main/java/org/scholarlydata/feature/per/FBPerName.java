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
public class FBPerName extends FeatureBuilderSPARQL<FeatureType, List<String>> {

    protected FeatureType type;
    protected Predicate predicate;

    public FBPerName(String sparqlEndpoint, FeatureType type, Predicate predicate){
        super(sparqlEndpoint);
        this.type=type;
        this.predicate=predicate;
    }

    @Override
    public Pair<FeatureType, List<String>> build(String objId) {
        String queryStr = SPARQLQueries.getObjectsOf(objId,
                predicate.getURI());
        ResultSet rs = query(queryStr);
        return new ImmutablePair<>(type, getListResult(rs));
    }
}
