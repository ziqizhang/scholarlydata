package org.scholarlydata.feature;

/**
 *
 */
public enum Predicate {

    ORGANIZATION_name("https://w3id.org/scholarlydata/ontology/conference-ontology.owl#name"),

    AFFLIATION_withOrganization("https://w3id.org/scholarlydata/ontology/conference-ontology.owl#withOrganisation"),
    AFFLIATION_isAffliationOf("https://w3id.org/scholarlydata/ontology/conference-ontology.owl#isAffiliationOf"),

    FUNCTION_during("https://w3id.org/scholarlydata/ontology/conference-ontology.owl#during"),

    PERSON_name("https://w3id.org/scholarlydata/ontology/conference-ontology.owl#name"),
    PERSON_givenName("https://w3id.org/scholarlydata/ontology/conference-ontology.owl#givenName"),
    PERSON_familyName("https://w3id.org/scholarlydata/ontology/conference-ontology.owl#familyName"),
    PERSON_holdsRole("https://w3id.org/scholarlydata/ontology/conference-ontology.owl#holdsRole"),
    PERSON_made("http://xmlns.com/foaf/0.1/made"),
    PERSON_hasAffliation("https://w3id.org/scholarlydata/ontology/conference-ontology.owl#hasAffiliation"),

    ROLE_withRole("https://w3id.org/scholarlydata/resource/participant");


    private String uri;

    Predicate(String uri){
        this.uri=uri;
    }

    public String getURI(){
        return uri;
    }
}
