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
public class FBPerAffliatedOrgName extends FeatureBuilderSPARQL<FeatureType, List<String>> {
    public FBPerAffliatedOrgName(String sparqlEndpoint) {
        super(sparqlEndpoint);
    }

    @Override
    public Pair<FeatureType, List<String>> build(String objId) {
        String queryStr = SPARQLQueries.pathObjObjObj(objId,
                Predicate.PERSON_hasAffliation.getURI(),
                Predicate.AFFLIATION_withOrganization.getURI(),
                Predicate.ORGANIZATION_name.getURI());

        ResultSet rs = query(queryStr);
        return new ImmutablePair<>(FeatureType.PERSON_AFFLIATED_ORGANIZATION_NAME, getListResult(rs));
    }
}
