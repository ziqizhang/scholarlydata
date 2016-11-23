package org.scholarlydata.feature.org;

import org.apache.commons.lang3.tuple.ImmutablePair;
import org.apache.commons.lang3.tuple.Pair;
import org.apache.jena.query.ResultSet;
import org.scholarlydata.SPARQLQueries;
import org.scholarlydata.feature.FeatureBuilderSPARQL;
import org.scholarlydata.feature.FeatureNormalizer;
import org.scholarlydata.feature.FeatureType;
import org.scholarlydata.feature.Predicate;

import java.util.List;

/**
 *
 */
public class FBOrgName extends FeatureBuilderSPARQL<FeatureType, List<String>> {

    public FBOrgName(String sparqlEndpoint) {
        super(sparqlEndpoint);
    }
    public FBOrgName(String sparqlEndpoint, FeatureNormalizer fn) {
        super(sparqlEndpoint,fn);
    }


    @Override
    public Pair<FeatureType, List<String>> build(String objectId) {
        String queryStr = SPARQLQueries.getObjectsOf(objectId,
                Predicate.ORGANIZATION_name.getURI());
        ResultSet rs = query(queryStr);
        return new ImmutablePair<>(FeatureType.ORGANIZATION_NAME, getListResult(rs));
    }
}
