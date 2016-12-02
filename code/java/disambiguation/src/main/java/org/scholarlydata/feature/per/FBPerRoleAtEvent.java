package org.scholarlydata.feature.per;

import org.apache.commons.lang3.tuple.ImmutablePair;
import org.apache.commons.lang3.tuple.Pair;
import org.apache.jena.query.QuerySolution;
import org.apache.jena.query.ResultSet;
import org.apache.jena.rdf.model.RDFNode;
import org.apache.jena.rdf.model.impl.LiteralImpl;
import org.scholarlydata.SPARQLQueries;
import org.scholarlydata.feature.FeatureBuilderSPARQL;
import org.scholarlydata.feature.FeatureType;
import org.scholarlydata.feature.Predicate;
import org.scholarlydata.util.SolrCache;

import java.util.ArrayList;
import java.util.List;

/**
 *
 */
public class FBPerRoleAtEvent extends FeatureBuilderSPARQL<FeatureType, List<String>> {
    public FBPerRoleAtEvent(String sparqlEndpoint,
                            SolrCache cache) {
        super(sparqlEndpoint,cache);
    }

    @Override
    public Pair<FeatureType, List<String>> build(String objId) {
        StringBuilder sb = new StringBuilder("select distinct ?e ?r where {\n");
        sb.append("<").append(objId).append("> <")
                .append(Predicate.PERSON_holdsRole.getURI())
                .append("> ")
                .append("?o . \n")
                .append("?o <").append(Predicate.ROLE_withRole.getURI()).append("> ?r . \n")
                .append("?o <").append(Predicate.FUNCTION_during.getURI()).append("> ?e .}");

        Object cached = getFromCache(sb.toString());
        if (cached != null)
            return new ImmutablePair<>(FeatureType.PERSON_ROLE_AT_EVENT_URI,
                    (List<String>) cached);

        ResultSet rs = query(sb.toString());
        List<String> out = new ArrayList<>();
        while (rs.hasNext()) {
            QuerySolution qs = rs.next();
            RDFNode event = qs.get("?e");
            RDFNode role = qs.get("?r");

            out.add(event.toString() + "|" +
                    role.toString());

        }

        saveToCache(sb.toString(), out);
        return new ImmutablePair<>(FeatureType.PERSON_ROLE_AT_EVENT_URI, out);

    }
}
