package org.scholarlydata.feature.org;

import org.apache.commons.lang3.tuple.ImmutablePair;
import org.apache.commons.lang3.tuple.Pair;
import org.apache.jena.query.ResultSet;
import org.scholarlydata.SPARQLQueries;
import org.scholarlydata.feature.FeatureBuilderSPARQL;
import org.scholarlydata.feature.FeatureNormalizer;
import org.scholarlydata.feature.FeatureType;
import org.scholarlydata.feature.Predicate;
import org.scholarlydata.util.SolrCache;

import java.util.List;

/**
 *
 */
public class FBOrgName extends FeatureBuilderSPARQL<FeatureType, List<String>> {

    public FBOrgName(String sparqlEndpoint, SolrCache cache) {
        super(sparqlEndpoint,cache);
    }
    public FBOrgName(String sparqlEndpoint, FeatureNormalizer fn,
                     SolrCache cache) {
        super(sparqlEndpoint,fn, cache);
    }


    @Override
    public Pair<FeatureType, List<String>> build(String objectId, boolean removeDuplicates) {
        String queryStr = SPARQLQueries.getObjectsOf(objectId,
                Predicate.ORGANIZATION_name.getURI());

        Object cached=getFromCache(queryStr);


        if(cached!=null) {
            List<String> result = (List<String>) cached;
            if(removeDuplicates)
                return new ImmutablePair<>(FeatureType.ORGANIZATION_NAME, removeDuplicates(result));
            else
                return new ImmutablePair<>(FeatureType.ORGANIZATION_NAME, result);
        }

        ResultSet rs = query(queryStr);
        List<String> result = getListResult(rs);
        if(removeDuplicates)
            result=removeDuplicates(result);
        saveToCache(queryStr, result);

        return new ImmutablePair<>(FeatureType.ORGANIZATION_NAME, result);
    }
}
