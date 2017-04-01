package org.scholarlydata.feature;

import org.apache.commons.lang.exception.ExceptionUtils;
import org.apache.jena.query.*;
import org.apache.jena.rdf.model.RDFNode;
import org.apache.log4j.Logger;
import org.scholarlydata.util.SolrCache;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 *
 */
public abstract class FeatureBuilderSPARQL<FeatureType, T> implements FeatureBuilder {

    protected Logger logger = Logger.getLogger(FeatureBuilderSPARQL.class.getName());
    protected String sparqlEndpoint;
    protected FeatureNormalizer normalizer;
    protected SolrCache cache;

    public FeatureBuilderSPARQL(String sparqlEndpoint,
                                SolrCache cache) {
        this.sparqlEndpoint = sparqlEndpoint;
        this.cache=cache;
    }
    public FeatureBuilderSPARQL(String sparqlEndpoint,
                                FeatureNormalizer fn,SolrCache cache){
        this(sparqlEndpoint, cache);
        this.normalizer=fn;
    }

    protected Object getFromCache(String queryString) {
        try {
            return cache.retrieve(queryString);
        } catch (Exception e) {
            logger.error(ExceptionUtils.getFullStackTrace(e));
        }
        return null;
    }

    protected void saveToCache(String queryString, Object obj){
        try {
            cache.cache(queryString, obj,true);
        } catch (Exception e) {
            logger.error(ExceptionUtils.getFullStackTrace(e));
        }
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

    protected List<String> removeDuplicates(List<String> in){
        Set<String> unique=new HashSet<>();
        for(String i: in){
            i=i.replaceAll("\\s+"," ").trim();
            unique.add(i);
        }
        return new ArrayList<>(unique);
    }
}
