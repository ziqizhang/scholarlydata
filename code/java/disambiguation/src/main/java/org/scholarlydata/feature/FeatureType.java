package org.scholarlydata.feature;

/**
 *
 */
public enum FeatureType {

    ORGANIZATION_NAME("org_name"), //name of the organisation
    ORGANIZATION_MEMBER_NAME("org_member_name"), //member's names, regardless of time
    ORGANISATION_MEMBER_URI("org_member_uri"), //member's uri, regardless of time
    ORGANISATION_PARTICIPATED_EVENT_URI("org_participated_event_uri"),//participated event uri


    PERSON_NAME("per_name"),
    PERSON_PARTICIPATED_EVENT_URI("per_participated_event_uri"),
    PERSON_AFFLIATED_ORGANIZATION_NAME("per_from_org_name"), //name of the person's organisation, regardless of time
    PERSON_AFFLIATED_ORGANIZATION_URI("per_from_org_uri"),
    PERSON_PUBLICATION_URI("per_published_work_uri"),
    PERSON_ROLE_AT_EVENT_URI("per_role_at_event_uri"), //pair composed of the uri of the role, and event
    PERSON_PUBLICATION_BOW("per_published_work_bow"), //bag of words of person publications based on abstract keyword and title
    PERSON_PUBLICATION_COAUTHOR("per_published_work_coauthor");


    private String name;

    FeatureType(String name){
        this.name=name;
    }

    public String getName(){
        return name;
    }
}
