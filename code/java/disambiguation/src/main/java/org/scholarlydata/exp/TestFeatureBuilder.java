package org.scholarlydata.exp;

import org.apache.commons.lang3.tuple.Pair;
import org.scholarlydata.feature.FeatureType;
import org.scholarlydata.feature.Predicate;
import org.scholarlydata.feature.org.FBOrgMemberName;
import org.scholarlydata.feature.org.FBOrgMemberURI;
import org.scholarlydata.feature.org.FBOrgParticipatedEventURI;
import org.scholarlydata.feature.per.*;

import java.util.List;

/**
 *
 */
public class TestFeatureBuilder {
    public static void main(String[] args) {
        String sparqlEndpoint = "http://www.scholarlydata.org/sparql/";
        //testOrgFeatureBuilder(sparqlEndpoint);
        testPerFeatureBuilder(sparqlEndpoint);
    }

    private static void testOrgFeatureBuilder(String sparqlEndpoint) {
        //1. provide object uris
        String[] uris = {"https://w3id.org/scholarlydata/organisation/shanghai-jiao-tong-university",
                "https://w3id.org/scholarlydata/organisation/shanghai-jiao-tong-university-china"};

        for (String uri : uris) {
            //2. create pipeline for feature generation
            Pair<FeatureType, List<String>> orgMemberNames = new FBOrgMemberName(sparqlEndpoint).build(uri);
            Pair<FeatureType, List<String>> orgMemberURI = new FBOrgMemberURI(sparqlEndpoint).build(uri);
            Pair<FeatureType, List<String>> orgName = new FBOrgMemberName(sparqlEndpoint).build(uri);
            Pair<FeatureType, List<String>> orgParticipatedEventURI = new FBOrgParticipatedEventURI(sparqlEndpoint).build(uri);
            System.out.println("end");
        }
    }

    private static void testPerFeatureBuilder(String sparqlEndpoint) {
        //1. provide object uris
        String[] uris = {"https://w3id.org/scholarlydata/person/raphaeel-troncy",
                "https://w3id.org/scholarlydata/person/raphael-troncy"};

        for (String uri : uris) {
            //2. create pipeline for feature generation
            Pair<FeatureType, List<String>> perAffliatedOrgName = new FBPerAffliatedOrgName(sparqlEndpoint).build(uri);
            Pair<FeatureType, List<String>> perAffliatedOrgURI = new FBPerAffliatedOrgURI(sparqlEndpoint).build(uri);
            Pair<FeatureType, List<String>> perName = new FBPerName(sparqlEndpoint,
                    FeatureType.PERSON_NAME, Predicate.PERSON_name).build(uri);
            Pair<FeatureType, List<String>> perGivenName = new FBPerName(sparqlEndpoint,
                    FeatureType.PERSON_GIVEN_NAME, Predicate.PERSON_givenName).build(uri);
            Pair<FeatureType, List<String>> perFamilyName = new FBPerName(sparqlEndpoint,
                    FeatureType.PERSON_FAMILY_NAME, Predicate.PERSON_familyName).build(uri);
            Pair<FeatureType, List<String>> perParticipatedEvent = new
                    FBPerParticipatedEventURI(sparqlEndpoint).build(uri);
            Pair<FeatureType, List<String>> perPublishedWorkURI = new
                    FBPerPublishedWorkURI(sparqlEndpoint).build(uri);
            Pair<FeatureType, List<Pair<String, String>>> perRoleAtEvent = new
                    FBPerRoleAtEvent(sparqlEndpoint).build(uri);
            System.out.println("end");
        }
    }
}
