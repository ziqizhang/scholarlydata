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
public class FBOrgMemberName extends FeatureBuilderSPARQL<FeatureType, List<String>> {
    public FBOrgMemberName(String sparqlEndpoint, SolrCache cache) {

        super(sparqlEndpoint, cache);
    }
    public FBOrgMemberName(String sparqlEndpoint, FeatureNormalizer fn,
                           SolrCache cache) {
        super(sparqlEndpoint,fn, cache);
    }

    @Override
    public Pair<FeatureType, List<String>> build(String objId) {

        String queryStr = SPARQLQueries.pathSubObjObj(objId,
                Predicate.AFFLIATION_withOrganization.getURI(),
                Predicate.AFFLIATION_isAffliationOf.getURI(),
                Predicate.PERSON_name.getURI());

        Object cached=getFromCache(queryStr);
        if(cached!=null)
            return new ImmutablePair<>(FeatureType.ORGANIZATION_MEMBER_NAME,(List<String>) cached);

        ResultSet rs = query(queryStr);
        List<String> result = getListResult(rs);
        saveToCache(queryStr, result);
        return new ImmutablePair<>(FeatureType.ORGANIZATION_MEMBER_NAME, result);
    }
}
