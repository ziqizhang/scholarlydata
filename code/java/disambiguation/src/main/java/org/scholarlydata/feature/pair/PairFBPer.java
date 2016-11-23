package org.scholarlydata.feature.pair;

import org.apache.commons.lang3.tuple.ImmutablePair;
import org.apache.commons.lang3.tuple.Pair;
import org.apache.log4j.Logger;
import org.scholarlydata.feature.FeatureNormalizer;
import org.scholarlydata.feature.FeatureType;
import org.scholarlydata.feature.Predicate;
import org.scholarlydata.feature.per.*;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Created by zqz on 23/11/16.
 */
public class PairFBPer implements PairFeatureBuilder {
    private static Logger log = Logger.getLogger(PairFBPer.class.getName());

    private String sparqlEndpoint;
    private Overlap[] overlapFunctions;
    private FeatureNormalizer fn = new FeatureNormalizer();

    public PairFBPer(String sparqlEndpoint){
        this.sparqlEndpoint=sparqlEndpoint;
        overlapFunctions = new Overlap[6];
        overlapFunctions[0]=new Overlap(0);
        overlapFunctions[1]=new Overlap(0, new SFSquareRoot());
        overlapFunctions[2]=new Overlap(1);
        overlapFunctions[3]=new Overlap(1, new SFSquareRoot());
        overlapFunctions[4]=new Overlap(2);
        overlapFunctions[5]=new Overlap(2, new SFSquareRoot());
    }
    @Override
    public Map<Pair<FeatureType, Integer>, Double> build(String obj1, String obj2) {

        log.info("Building features for: "+obj1);
        log.info("\t"+FBPerAffliatedOrgName.class.getCanonicalName());
        Pair<FeatureType, List<String>> perAffOrgName1 = new FBPerAffliatedOrgName(sparqlEndpoint, fn).build(obj1);
        log.info("\t"+FBPerAffliatedOrgURI.class.getCanonicalName());
        Pair<FeatureType, List<String>> perAffOrgURI1 = new FBPerAffliatedOrgURI(sparqlEndpoint).build(obj1);
        log.info("\t"+FBPerName.class.getCanonicalName());
        Pair<FeatureType, List<String>> perName1 = new FBPerName(sparqlEndpoint,
                FeatureType.PERSON_NAME,Predicate.PERSON_name, fn).build(obj1);
        log.info("\t"+FBPerParticipatedEventURI.class.getCanonicalName());
        Pair<FeatureType, List<String>> perParticipatedEventURI1 = new FBPerParticipatedEventURI(sparqlEndpoint).build(obj1);
        log.info("\t"+ FBPerPublishedWorkURI.class.getCanonicalName());
        Pair<FeatureType, List<String>> perPublishedWorkURI1 = new FBPerPublishedWorkURI(sparqlEndpoint).build(obj1);
        log.info("\t"+ FBPerRoleAtEvent.class.getCanonicalName());
        Pair<FeatureType, List<String>> perRoleAtEvent1 = new FBPerRoleAtEvent(sparqlEndpoint).build(obj1);

        log.info("Building features for: "+obj2);
        log.info("\t"+FBPerAffliatedOrgName.class.getCanonicalName());
        Pair<FeatureType, List<String>> perAffOrgName2 = new FBPerAffliatedOrgName(sparqlEndpoint, fn).build(obj2);
        log.info("\t"+FBPerAffliatedOrgURI.class.getCanonicalName());
        Pair<FeatureType, List<String>> perAffOrgURI2 = new FBPerAffliatedOrgURI(sparqlEndpoint).build(obj2);
        log.info("\t"+FBPerName.class.getCanonicalName());
        Pair<FeatureType, List<String>> perName2 = new FBPerName(sparqlEndpoint,
                FeatureType.PERSON_NAME,Predicate.PERSON_name, fn).build(obj2);
        log.info("\t"+FBPerParticipatedEventURI.class.getCanonicalName());
        Pair<FeatureType, List<String>> perParticipatedEventURI2 = new FBPerParticipatedEventURI(sparqlEndpoint).build(obj2);
        log.info("\t"+ FBPerPublishedWorkURI.class.getCanonicalName());
        Pair<FeatureType, List<String>> perPublishedWorkURI2 = new FBPerPublishedWorkURI(sparqlEndpoint).build(obj2);
        log.info("\t"+ FBPerRoleAtEvent.class.getCanonicalName());
        Pair<FeatureType, List<String>> perRoleAtEvent2 = new FBPerRoleAtEvent(sparqlEndpoint).build(obj2);

        Map<Pair<FeatureType, Integer>, Double> result = new HashMap<>();
        generateOverlapFeatures(result, perAffOrgName1.getValue(), perAffOrgName2.getValue(), perAffOrgName1.getKey());
        generateOverlapFeatures(result, perAffOrgURI1.getValue(), perAffOrgURI2.getValue(), perAffOrgURI1.getKey());
        generateOverlapFeatures(result, perName1.getValue(), perName2.getValue(), perName1.getKey());
        generateOverlapFeatures(result, perParticipatedEventURI1.getValue(), perParticipatedEventURI2.getValue(), perParticipatedEventURI1.getKey());
        generateOverlapFeatures(result, perPublishedWorkURI1.getValue(), perPublishedWorkURI2.getValue(), perPublishedWorkURI1.getKey());
        generateOverlapFeatures(result, perRoleAtEvent1.getValue(), perRoleAtEvent2.getValue(), perRoleAtEvent1.getKey());

        return result;
    }

    protected void generateOverlapFeatures(Map<Pair<FeatureType, Integer>, Double> result,
                                           List<String> obj1,
                                           List<String> obj2, FeatureType ft){
        for(Overlap of : overlapFunctions){
            double score = of.score(obj1, obj2);
            result.put(new ImmutablePair<>(ft, of.getOption()), score);
        }
    }
}
