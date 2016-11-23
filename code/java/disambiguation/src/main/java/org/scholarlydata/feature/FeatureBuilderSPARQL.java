package org.scholarlydata.feature;

import org.apache.jena.query.*;
import org.apache.jena.rdf.model.RDFNode;
import org.apache.jena.rdf.model.impl.LiteralImpl;

import java.util.ArrayList;
import java.util.List;

/**
 *
 */
public abstract class FeatureBuilderSPARQL<FeatureType, T> implements FeatureBuilder {

    protected String sparqlEndpoint;
    protected FeatureNormalizer normalizer;

    public FeatureBuilderSPARQL(String sparqlEndpoint) {
        this.sparqlEndpoint = sparqlEndpoint;
    }
    public FeatureBuilderSPARQL(String sparqlEndpoint, FeatureNormalizer fn){
        this(sparqlEndpoint);
        this.normalizer=fn;
    }

    protected ResultSet query(String queryString) {
        org.apache.jena.query.Query query = QueryFactory.create(queryString);
        QueryExecution qexec = QueryExecutionFactory.sparqlService(sparqlEndpoint, query);

        ResultSet rs = qexec.execSelect();
        return rs;
    }

    protected List<String> getListResult(ResultSet rs) {
        List<String> out = new ArrayList<>();
        while (rs.hasNext()) {
            QuerySolution qs = rs.next();
            RDFNode range = qs.get("?o");
            if(normalizer!=null)
                out.add(normalizer.normalize(range.toString()));
            else
                out.add(range.toString().toLowerCase());
        }
        return out;
    }
}
