package org.scholarlydata.feature.per;

import org.apache.commons.lang3.tuple.ImmutablePair;
import org.apache.commons.lang3.tuple.Pair;
import org.apache.jena.query.ResultSet;
import org.scholarlydata.SPARQLQueries;
import org.scholarlydata.feature.FeatureBuilderSPARQL;
import org.scholarlydata.feature.FeatureType;
import org.scholarlydata.feature.Predicate;
import org.scholarlydata.util.SolrCache;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 *
 */
public class FBPerPublishedWorkURI extends FeatureBuilderSPARQL<FeatureType, List<String>> {
    public FBPerPublishedWorkURI(String sparqlEndpoint,
                                 SolrCache cache) {
        super(sparqlEndpoint, cache);
    }

    @Override
    public Pair<FeatureType, List<String>> build(String objId) {
        String queryStr = SPARQLQueries.getObjectsOf(objId,
                Predicate.PERSON_made.getURI());

        Object cached = getFromCache(queryStr);
        if (cached != null)
            return new ImmutablePair<>(FeatureType.PERSON_PUBLICATION_URI,
                    (List<String>) cached);

        ResultSet rs = query(queryStr);
        Set<String> publications = new HashSet<>(getListResult(rs));

        StringBuilder sb = new StringBuilder("select distinct ?o where {\n ?s <");
        sb.append(Predicate.AUTHOR_lIST_ITEM_hasContent.getURI()).append("> <")
                .append(objId).append("> .\n?l <")
                .append(Predicate.AUTHOR_LIST_hasItem.getURI()).append("> ?s .\n?o <")
                .append(Predicate.PUBLICATION_hasAuthorList.getURI()).append("> ?l .\n}");
        rs = query(sb.toString());
        publications.addAll(getListResult(rs));

        saveToCache(queryStr, new ArrayList<>(publications));
        return new ImmutablePair<>(FeatureType.PERSON_PUBLICATION_URI, new ArrayList<>(publications));
    }
}
