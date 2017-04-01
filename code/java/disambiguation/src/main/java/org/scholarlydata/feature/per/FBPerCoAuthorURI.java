package org.scholarlydata.feature.per;

import org.apache.commons.lang3.tuple.ImmutablePair;
import org.apache.commons.lang3.tuple.Pair;
import org.apache.jena.query.ResultSet;
import org.scholarlydata.feature.FeatureBuilderSPARQL;
import org.scholarlydata.feature.FeatureType;
import org.scholarlydata.feature.Predicate;
import org.scholarlydata.util.SolrCache;

import java.util.List;

/**
 *
 */
public class FBPerCoAuthorURI extends FeatureBuilderSPARQL<FeatureType, List<String>> {
    public FBPerCoAuthorURI(String sparqlEndpoint,
                            SolrCache cache) {
        super(sparqlEndpoint, cache);
    }

    @Override
    public Pair<FeatureType, List<String>> build(String objId, boolean removeDuplicates) {
        StringBuilder sb = new StringBuilder("select distinct ?o where {\n ?li <");
        sb.append(Predicate.AUTHOR_lIST_ITEM_hasContent.getURI()).append("> <")
                .append(objId).append("> .\n?al <")
                .append(Predicate.AUTHOR_LIST_hasItem.getURI()).append("> ?li .\n?al <")
                .append(Predicate.AUTHOR_LIST_hasItem.getURI()).append("> ?lis .\n?lis <")
                .append(Predicate.AUTHOR_lIST_ITEM_hasContent.getURI()).append("> ?o .}");

        Object cached = getFromCache(sb.toString());
        if (cached != null) {
            List<String> result = (List<String>) cached;
            if (removeDuplicates)
                return new ImmutablePair<>(FeatureType.PERSON_PUBLICATION_COAUTHOR,
                        removeDuplicates(result));
            else
                return new ImmutablePair<>(FeatureType.PERSON_PUBLICATION_COAUTHOR,
                        result);
        }

        ResultSet rs = query(sb.toString());
        List<String> out = getListResult(rs);
        if (removeDuplicates)
            out = removeDuplicates(out);
        out.remove(objId);
        saveToCache(sb.toString(), out);
        return new ImmutablePair<>(FeatureType.PERSON_PUBLICATION_COAUTHOR, out);


    }
}
