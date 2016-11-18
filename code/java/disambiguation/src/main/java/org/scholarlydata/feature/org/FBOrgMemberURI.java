package org.scholarlydata.feature.org;

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
public class FBOrgMemberURI extends FeatureBuilderSPARQL<FeatureType, List<String>> {

    @Override
    public Pair<FeatureType, List<String>> build(String objId) {

        String queryStr = SPARQLQueries.pathSubObj(objId,
                Predicate.AFFLIATION_withOrganization.getURI(),
                Predicate.AFFLIATION_isAffliationOf.getURI());

        ResultSet rs = query(queryStr);
        return new ImmutablePair<>(FeatureType.ORGANISATION_MEMBER_URI, getListResult(rs));
    }
}
