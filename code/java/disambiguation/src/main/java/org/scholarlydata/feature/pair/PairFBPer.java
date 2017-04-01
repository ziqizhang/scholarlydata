package org.scholarlydata.feature.pair;

import org.apache.commons.lang3.tuple.ImmutablePair;
import org.apache.commons.lang3.tuple.Pair;
import org.apache.log4j.Logger;
import org.scholarlydata.exp.FeatureGenerator;
import org.scholarlydata.feature.FeatureNormalizer;
import org.scholarlydata.feature.FeatureType;
import org.scholarlydata.feature.Predicate;
import org.scholarlydata.feature.per.*;
import org.scholarlydata.util.SolrCache;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Created by zqz on 23/11/16.
 */
public class PairFBPer implements PairFeatureBuilder {
    private static Logger log = Logger.getLogger(PairFBPer.class.getName());
    private final boolean USE_PRESENCE_FEATURE=true;
    private final boolean USE_EQUATRE_NORM=false;
    private final boolean USE_STRING_SIMILARITY=false;

    private String sparqlEndpoint;
    private SimilarityMeasure[] generalSimilarityMeasures;
    private MultiSetOverlap[] multiSetOverlapFunctions;
    private FeatureNormalizer fn = new FeatureNormalizer();
    private List<String> stopwords;
    private SolrCache cache;

    public PairFBPer(String sparqlEndpoint, List<String> stopwords, SolrCache cache){
        this.sparqlEndpoint=sparqlEndpoint;
        if(USE_STRING_SIMILARITY){
            generalSimilarityMeasures = new SimilarityMeasure[6];
            generalSimilarityMeasures[4]=new StringSim("lev",false);
            //generalSimilarityMeasures[5]=new StringSim("lev",false, new SFSquareRoot());
        }
        else
            generalSimilarityMeasures = new SimilarityMeasure[4];
        //generalSimilarityMeasures = new SetOverlap[4];
        generalSimilarityMeasures[0]=new SetOverlap("union",USE_EQUATRE_NORM);
        generalSimilarityMeasures[1]=new SetOverlap("union", USE_EQUATRE_NORM,new SFSquareRoot());
        generalSimilarityMeasures[2]=new SetOverlap("max", USE_EQUATRE_NORM);
        generalSimilarityMeasures[3]=new SetOverlap("max", USE_EQUATRE_NORM,new SFSquareRoot());
        //generalSimilarityMeasures[4]=new SetOverlap(2);
        //generalSimilarityMeasures[5]=new SetOverlap(2, new SFSquareRoot());

        this.multiSetOverlapFunctions=new MultiSetOverlap[4];
        multiSetOverlapFunctions[0]=new MultiSetOverlap("union",USE_EQUATRE_NORM);
        multiSetOverlapFunctions[1]=new MultiSetOverlap("union",USE_EQUATRE_NORM, new SFSquareRoot());
        multiSetOverlapFunctions[2]=new MultiSetOverlap("max",USE_EQUATRE_NORM);
        multiSetOverlapFunctions[3]=new MultiSetOverlap("max",USE_EQUATRE_NORM, new SFSquareRoot());
        this.stopwords=stopwords;

        this.cache=cache;
    }
    @Override
    public Map<Pair<FeatureType, String>, Double> build(String obj1, String obj2) {

        log.info("Building features for: "+obj1);
        log.info("\t"+FBPerAffliatedOrgName.class.getCanonicalName());
        Pair<FeatureType, List<String>> perAffOrgName1 = new FBPerAffliatedOrgName(sparqlEndpoint, fn, cache).build(obj1,true);
        log.info("\t"+FBPerAffliatedOrgURI.class.getCanonicalName());
        Pair<FeatureType, List<String>> perAffOrgURI1 = new FBPerAffliatedOrgURI(sparqlEndpoint, cache).build(obj1,true);
        log.info("\t"+FBPerName.class.getCanonicalName());
        Pair<FeatureType, List<String>> perName1 = new FBPerName(sparqlEndpoint,
                FeatureType.PERSON_NAME,Predicate.PERSON_name, fn, cache).build(obj1,true);
        log.info("\t"+FBPerParticipatedEventURI.class.getCanonicalName());
        Pair<FeatureType, List<String>> perParticipatedEventURI1 = new FBPerParticipatedEventURI(sparqlEndpoint, cache).build(obj1,true);
        log.info("\t"+ FBPerPublishedWorkURI.class.getCanonicalName());
        Pair<FeatureType, List<String>> perPublishedWorkURI1 = new FBPerPublishedWorkURI(sparqlEndpoint, cache).build(obj1,true);
        log.info("\t"+ FBPerRoleAtEvent.class.getCanonicalName());
        Pair<FeatureType, List<String>> perRoleAtEvent1 = new FBPerRoleAtEvent(sparqlEndpoint, cache).build(obj1,true);
        log.info("\t"+FBPerCoAuthorURI.class.getCanonicalName());
        Pair<FeatureType, List<String>> perCoAuthor1=new FBPerCoAuthorURI(sparqlEndpoint, cache).build(obj1,false);
        log.info("\t"+FBPerPublishedWorkKAT.class.getCanonicalName());
        Pair<FeatureType, List<String>> perKAT1 = new FBPerPublishedWorkKAT(sparqlEndpoint,fn,stopwords, cache).build(obj1,false);

        log.info("Building features for: "+obj2);
        log.info("\t"+FBPerAffliatedOrgName.class.getCanonicalName());
        Pair<FeatureType, List<String>> perAffOrgName2 = new FBPerAffliatedOrgName(sparqlEndpoint, fn, cache).build(obj2,true);
        log.info("\t"+FBPerAffliatedOrgURI.class.getCanonicalName());
        Pair<FeatureType, List<String>> perAffOrgURI2 = new FBPerAffliatedOrgURI(sparqlEndpoint, cache).build(obj2,true);
        log.info("\t"+FBPerName.class.getCanonicalName());
        Pair<FeatureType, List<String>> perName2 = new FBPerName(sparqlEndpoint,
                FeatureType.PERSON_NAME,Predicate.PERSON_name, fn, cache).build(obj2,true);
        log.info("\t"+FBPerParticipatedEventURI.class.getCanonicalName());
        Pair<FeatureType, List<String>> perParticipatedEventURI2 = new FBPerParticipatedEventURI(sparqlEndpoint, cache).build(obj2,true);
        log.info("\t"+ FBPerPublishedWorkURI.class.getCanonicalName());
        Pair<FeatureType, List<String>> perPublishedWorkURI2 = new FBPerPublishedWorkURI(sparqlEndpoint, cache).build(obj2,true);
        log.info("\t"+ FBPerRoleAtEvent.class.getCanonicalName());
        Pair<FeatureType, List<String>> perRoleAtEvent2 = new FBPerRoleAtEvent(sparqlEndpoint, cache).build(obj2,true);
        log.info("\t"+FBPerCoAuthorURI.class.getCanonicalName());
        Pair<FeatureType, List<String>> perCoAuthor2=new FBPerCoAuthorURI(sparqlEndpoint, cache).build(obj2,false);
        log.info("\t"+FBPerPublishedWorkKAT.class.getCanonicalName());
        Pair<FeatureType, List<String>> perKAT2 = new FBPerPublishedWorkKAT(sparqlEndpoint,fn,stopwords, cache).build(obj2,false);

        Map<Pair<FeatureType, String>, Double> result = new LinkedHashMap<>();
        generateSimilarityFeatures(result, perAffOrgName1.getValue(), perAffOrgName2.getValue(), perAffOrgName1.getKey());
        generateSimilarityFeatures(result, perAffOrgURI1.getValue(), perAffOrgURI2.getValue(), perAffOrgURI1.getKey());
        generateSimilarityFeatures(result, perName1.getValue(), perName2.getValue(), perName1.getKey());
        generateSimilarityFeatures(result, perParticipatedEventURI1.getValue(), perParticipatedEventURI2.getValue(), perParticipatedEventURI1.getKey());
        generateSimilarityFeatures(result, perPublishedWorkURI1.getValue(), perPublishedWorkURI2.getValue(), perPublishedWorkURI1.getKey());
        generateSimilarityFeatures(result, perRoleAtEvent1.getValue(), perRoleAtEvent2.getValue(), perRoleAtEvent1.getKey());
        generateSimilarityFeatures(result, perCoAuthor1.getValue(), perCoAuthor2.getValue(), perCoAuthor1.getKey());
        generateOverlapKAT(result, perKAT1.getValue(), perKAT2.getValue(), perKAT1.getKey());

        return result;
    }

    private void generateOverlapKAT(Map<Pair<FeatureType, String>, Double> result,
                                    List<String> obj1,
                                    List<String> obj2, FeatureType ft) {
        for(MultiSetOverlap of : multiSetOverlapFunctions){
            if(of==null)
                continue;
            double score = of.score(obj1, obj2);
            if(Double.isNaN(score)||Double.isInfinite(score))
                score=0.0;
            result.put(new ImmutablePair<>(ft, of.getOption()+"|"+of.getSf()), score);

            if(USE_PRESENCE_FEATURE){
                FeatureGenerator.generatePresenceFeature(result, obj1, obj2, ft);
            }
        }
    }

    protected void generateSimilarityFeatures(Map<Pair<FeatureType, String>, Double> result,
                                              List<String> obj1,
                                              List<String> obj2, FeatureType ft){
        for(SimilarityMeasure of : generalSimilarityMeasures){
            if(of==null)
                continue;
            double score = of.score(obj1, obj2);
            if(Double.isNaN(score)||Double.isInfinite(score))
                score=0.0;
            result.put(new ImmutablePair<>(ft, of.getOption()+"|"+of.getSf()), score);
            if(USE_PRESENCE_FEATURE){
                FeatureGenerator.generatePresenceFeature(result, obj1, obj2, ft);
            }
        }
    }
}
