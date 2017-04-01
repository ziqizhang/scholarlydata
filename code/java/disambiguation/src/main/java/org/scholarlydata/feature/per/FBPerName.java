package org.scholarlydata.feature.per;

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
public class FBPerName extends FeatureBuilderSPARQL<FeatureType, List<String>> {

    protected FeatureType type;
    protected Predicate predicate;

    public FBPerName(String sparqlEndpoint, FeatureType type, Predicate predicate,
                     SolrCache cache){
        super(sparqlEndpoint, cache);
        this.type=type;
        this.predicate=predicate;
    }

    public FBPerName(String sparqlEndpoint, FeatureType type, Predicate predicate,
                     FeatureNormalizer fn, SolrCache cache){
        super(sparqlEndpoint,fn, cache);
        this.type=type;
        this.predicate=predicate;
    }

    @Override
    public Pair<FeatureType, List<String>> build(String objId, boolean removeDuplicates) {
        String queryStr = SPARQLQueries.getObjectsOf(objId,
                predicate.getURI());

        Object cached = getFromCache(queryStr);
        if (cached != null) {
            List<String> result = (List<String>) cached;

            if(removeDuplicates)
                return new ImmutablePair<>(type,
                        removeDuplicates(result));
            else
                return new ImmutablePair<>(type,
                        result);
        }

        ResultSet rs = query(queryStr);
        List<String> result = getListResult(rs);
        if(removeDuplicates)
            result=removeDuplicates(result);
        saveToCache(queryStr, result);
        return new ImmutablePair<>(type, result);
    }
}
