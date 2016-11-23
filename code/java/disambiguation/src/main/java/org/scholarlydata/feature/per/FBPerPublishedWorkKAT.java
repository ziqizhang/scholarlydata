package org.scholarlydata.feature.per;

import org.apache.commons.lang3.tuple.ImmutablePair;
import org.apache.commons.lang3.tuple.Pair;
import org.apache.jena.query.QuerySolution;
import org.apache.jena.query.ResultSet;
import org.apache.jena.rdf.model.RDFNode;
import org.scholarlydata.feature.FeatureBuilderSPARQL;
import org.scholarlydata.feature.FeatureNormalizer;
import org.scholarlydata.feature.FeatureType;
import org.scholarlydata.feature.Predicate;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * published work's keywords, abstract, and title
 */
public class FBPerPublishedWorkKAT extends FeatureBuilderSPARQL<FeatureType, List<String>> {
    protected List<String> stopwords;

    public FBPerPublishedWorkKAT(String sparqlEndpoint) {
        super(sparqlEndpoint);
    }

    public FBPerPublishedWorkKAT(String sparqlEndpoint, FeatureNormalizer normalizer,List<String> stopwords){
        super(sparqlEndpoint, normalizer);
        this.stopwords=stopwords;

    }

    @Override
    public Pair<FeatureType, List<String>> build(String objId) {

        StringBuilder sb = new StringBuilder("select distinct ?k ?a ?t where {\n ?s <");
        sb.append(Predicate.AUTHOR_lIST_ITEM_hasContent.getURI()).append("> <")
                .append(objId).append("> .\n?l <")
                .append(Predicate.AUTHOR_LIST_hasItem.getURI()).append("> ?s .\n?o <")
                .append(Predicate.PUBLICATION_hasAuthorList.getURI()).append("> ?l .\n{?o <")
                .append(Predicate.PUBLICATION_hasAbstract.getURI()).append("> ?a .}\n union {?o <")
                .append(Predicate.PUBLICATION_hasTitle.getURI()).append("> ?t .}\n union {?o <")
                .append(Predicate.PUBLICATION_hasKeyword.getURI()).append("> ?k .}}")
        ;
        ResultSet rs = query(sb.toString());

        List<String> out = new ArrayList<>();

        Set<String> uniqueValues = new HashSet<>();
        while (rs.hasNext()) {
            QuerySolution qs = rs.next();
            RDFNode keywords = qs.get("?k");
            RDFNode abstracts = qs.get("?a");
            RDFNode title = qs.get("?t");
            if(keywords!=null)
                uniqueValues.add(keywords.toString());
            if(abstracts!=null)
                uniqueValues.add(abstracts.toString());
            if(title!=null)
                uniqueValues.add(title.toString());
        }

        for(String v: uniqueValues) {
            splitAndAdd(v, out);
        }

        if(stopwords!=null)
            out.removeAll(stopwords);

        return new ImmutablePair<>(FeatureType.PERSON_PUBLICATION_BOW, out);
    }

    protected void splitAndAdd(String string, List<String> out) {
        if (normalizer != null) {
            for (String s : normalizer.normalize(string).split("\\s+")) {
                if (s.length() > 0) {
                    out.add(s);
                }
            }
        } else {
            for (String s : string.toLowerCase().split("\\s+")) {
                if (s.length() > 0) {
                    out.add(s);
                }
            }
        }
    }
}
