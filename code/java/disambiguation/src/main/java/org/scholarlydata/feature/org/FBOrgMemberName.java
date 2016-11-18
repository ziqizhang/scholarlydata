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
public class FBOrgMemberName extends FeatureBuilderSPARQL<FeatureType, List<String>> {
    @Override
    public Pair<FeatureType, List<String>> build(String objId) {

        String queryStr = SPARQLQueries.pathSubObjObj(objId,
                Predicate.AFFLIATION_withOrganization.getURI(),
                Predicate.AFFLIATION_isAffliationOf.getURI(),
                Predicate.PERSON_name.getURI());


        ResultSet rs = query(queryStr);
        return new ImmutablePair<>(FeatureType.ORGANIZATION_MEMBER_NAME, getListResult(rs));
    }
}
