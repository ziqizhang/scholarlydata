package org.scholarlydata.feature.pair;

import org.apache.commons.lang3.tuple.ImmutablePair;
import org.apache.commons.lang3.tuple.Pair;
import org.apache.log4j.Logger;
import org.scholarlydata.feature.FeatureNormalizer;
import org.scholarlydata.feature.FeatureType;
import org.scholarlydata.feature.org.FBOrgMemberName;
import org.scholarlydata.feature.org.FBOrgMemberURI;
import org.scholarlydata.feature.org.FBOrgName;
import org.scholarlydata.feature.org.FBOrgParticipatedEventURI;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 *
 */
public class PairFBOrg implements PairFeatureBuilder {
    private static Logger log = Logger.getLogger(PairFBOrg.class.getName());

    private String sparqlEndpoint;
    private Overlap[] overlapFunctions;
    private FeatureNormalizer fn = new FeatureNormalizer();

    public PairFBOrg(String sparqlEndpoint){
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
        log.info("\t"+FBOrgName.class.getCanonicalName());
        Pair<FeatureType, List<String>> orgName1 = new FBOrgName(sparqlEndpoint, fn).build(obj1);
        log.info("\t"+FBOrgMemberURI.class.getCanonicalName());
        Pair<FeatureType, List<String>> orgMemberURI1 = new FBOrgMemberURI(sparqlEndpoint).build(obj1);
        log.info("\t"+FBOrgMemberName.class.getCanonicalName());
        Pair<FeatureType, List<String>> orgMemberName1 = new FBOrgMemberName(sparqlEndpoint,fn).build(obj1);
        log.info("\t"+FBOrgParticipatedEventURI.class.getCanonicalName());
        Pair<FeatureType, List<String>> orgParticipatedEventURI1 = new FBOrgParticipatedEventURI(sparqlEndpoint).build(obj1);

        log.info("Building features for: "+obj2);
        log.info("\t"+FBOrgName.class.getCanonicalName());
        Pair<FeatureType, List<String>> orgName2 = new FBOrgName(sparqlEndpoint, fn).build(obj2);
        log.info("\t"+FBOrgMemberURI.class.getCanonicalName());
        Pair<FeatureType, List<String>> orgMemberURI2 = new FBOrgMemberURI(sparqlEndpoint).build(obj2);
        log.info("\t"+FBOrgMemberName.class.getCanonicalName());
        Pair<FeatureType, List<String>> orgMemberName2 = new FBOrgMemberName(sparqlEndpoint,fn).build(obj2);
        log.info("\t"+FBOrgParticipatedEventURI.class.getCanonicalName());
        Pair<FeatureType, List<String>> orgParticipatedEventURI2 = new FBOrgParticipatedEventURI(sparqlEndpoint).build(obj2);

        Map<Pair<FeatureType, Integer>, Double> result = new HashMap<>();
        generateOverlapFeatures(result, orgName1.getValue(), orgName2.getValue(), orgName1.getKey());
        generateOverlapFeatures(result, orgMemberName1.getValue(), orgMemberName2.getValue(), orgMemberName1.getKey());
        generateOverlapFeatures(result, orgMemberURI1.getValue(), orgMemberURI2.getValue(), orgMemberURI1.getKey());
        generateOverlapFeatures(result, orgParticipatedEventURI1.getValue(), orgParticipatedEventURI2.getValue(), orgParticipatedEventURI1.getKey());

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
