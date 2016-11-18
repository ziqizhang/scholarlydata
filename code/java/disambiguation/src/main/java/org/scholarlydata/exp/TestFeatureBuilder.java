package org.scholarlydata.exp;

import org.apache.commons.lang3.tuple.Pair;
import org.scholarlydata.feature.FeatureType;
import org.scholarlydata.feature.org.FBOrgMemberName;
import org.scholarlydata.feature.org.FBOrgMemberURI;
import org.scholarlydata.feature.org.FBOrgParticipatedEventURI;

import java.util.List;

/**
 * Created by zqz on 18/11/16.
 */
public class TestFeatureBuilder {
    public static void main(String[] args) {
        String sparqlEndpoint="http://www.scholarlydata.org/sparql/";
        //1. provide object uris
        String[] uris = {"http://data.semanticweb.org/organization/shanghai-jiao-tong-university",
                "http://data.semanticweb.org/organization/shanghai-jiao-tong-university-china"};

        for(String uri: uris) {
            //2. create pipeline for feature generation
            Pair<FeatureType, List<String>> orgMemberNames = new FBOrgMemberName(sparqlEndpoint).build(uri);
            Pair<FeatureType, List<String>> orgMemberURI = new FBOrgMemberURI(sparqlEndpoint).build(uri);
            Pair<FeatureType, List<String>> orgName = new FBOrgMemberName(sparqlEndpoint).build(uri);
            Pair<FeatureType, List<String>> orgParticipatedEventURI = new FBOrgParticipatedEventURI(sparqlEndpoint).build(uri);
        }

    }
}
