package org.scholarlydata.feature.per;

import org.apache.commons.lang3.tuple.ImmutablePair;
import org.apache.commons.lang3.tuple.Pair;
import org.apache.jena.query.ResultSet;
import org.scholarlydata.SPARQLQueries;
import org.scholarlydata.feature.FeatureBuilderSPARQL;
import org.scholarlydata.feature.FeatureType;
import org.scholarlydata.feature.Predicate;
import org.scholarlydata.util.SolrCache;

import java.util.List;

/**
 *
 */
public class FBPerAffliatedOrgURI extends FeatureBuilderSPARQL<FeatureType, List<String>> {
    public FBPerAffliatedOrgURI(String sparqlEndpoint,
                                SolrCache cache) {
        super(sparqlEndpoint, cache);
    }

    @Override
    public Pair<FeatureType, List<String>> build(String objId, boolean removeDuplicates) {
        String queryStr = SPARQLQueries.pathObjObj(objId,
                Predicate.PERSON_hasAffliation.getURI(),
                Predicate.AFFLIATION_withOrganization.getURI());

        Object cached = getFromCache(queryStr);
        if (cached != null) {
            List<String> result = (List<String>) cached;

            if(removeDuplicates)
                return new ImmutablePair<>(FeatureType.PERSON_AFFLIATED_ORGANIZATION_URI,
                        removeDuplicates(result));
            else
                return new ImmutablePair<>(FeatureType.PERSON_AFFLIATED_ORGANIZATION_URI,
                        result);
        }

        ResultSet rs = query(queryStr);
        List<String> result = getListResult(rs);
        if(removeDuplicates)
            result=removeDuplicates(result);
        saveToCache(queryStr, result);
        return new ImmutablePair<>(FeatureType.PERSON_AFFLIATED_ORGANIZATION_URI, result);
    }
}
