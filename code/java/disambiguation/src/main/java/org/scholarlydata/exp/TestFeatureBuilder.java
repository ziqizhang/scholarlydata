package org.scholarlydata.exp;

import org.apache.commons.lang3.tuple.Pair;
import org.scholarlydata.feature.FeatureType;
import org.scholarlydata.feature.pair.PairFBOrg;
import org.scholarlydata.feature.pair.PairFBPer;

import java.util.ArrayList;
import java.util.Map;

/**
 *
 */
public class TestFeatureBuilder {
    public static void main(String[] args) {
        String sparqlEndpoint = "http://www.scholarlydata.org/sparql/";
        testOrgFeatureBuilder(sparqlEndpoint);
        //testPerFeatureBuilder(sparqlEndpoint);
    }

    private static void testOrgFeatureBuilder(String sparqlEndpoint) {
        //1. provide object uris
        String obj1 = "https://w3id.org/scholarlydata/organisation/shanghai-jiao-tong-university";
        String obj2 = "https://w3id.org/scholarlydata/organisation/shanghai-jiao-tong-university-china";
        PairFBOrg fb = new PairFBOrg(sparqlEndpoint);
        Map<Pair<FeatureType, String>, Double> features = fb.build(obj1, obj2);
        System.out.println("end");
    }

    private static void testPerFeatureBuilder(String sparqlEndpoint) {
        //1. provide object uris
        String obj1 = "https://w3id.org/scholarlydata/person/raphaeel-troncy";
        String obj2 = "https://w3id.org/scholarlydata/person/raphael-troncy";
        PairFBPer fb = new PairFBPer(sparqlEndpoint, new ArrayList<>());
        Map<Pair<FeatureType, String>, Double> features = fb.build(obj1, obj2);
        System.out.println("end");
    }
}
