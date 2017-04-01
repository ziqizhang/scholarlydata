package org.scholarlydata.feature.pair;

import org.apache.commons.lang3.tuple.ImmutablePair;
import org.apache.commons.lang3.tuple.Pair;
import org.apache.log4j.Logger;
import org.scholarlydata.exp.FeatureGenerator;
import org.scholarlydata.feature.FeatureNormalizer;
import org.scholarlydata.feature.FeatureType;
import org.scholarlydata.feature.org.FBOrgMemberName;
import org.scholarlydata.feature.org.FBOrgMemberURI;
import org.scholarlydata.feature.org.FBOrgName;
import org.scholarlydata.feature.org.FBOrgParticipatedEventURI;
import org.scholarlydata.util.SolrCache;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 *
 */
public class PairFBOrg implements PairFeatureBuilder {
	private static Logger log = Logger.getLogger(PairFBOrg.class.getName());
	private final boolean USE_PRESENCE_FEATURE=true;
	private final boolean USE_EQUATRE_NORM=true;
    private final boolean USE_STRING_SIMILARITY=true;
	private String sparqlEndpoint;
	private SimilarityMeasure[] similarityMeasures;
	private FeatureNormalizer fn = new FeatureNormalizer();
	private SolrCache cache;

	public PairFBOrg(String sparqlEndpoint, SolrCache cache){
		this.sparqlEndpoint=sparqlEndpoint;
        if(USE_STRING_SIMILARITY){
            similarityMeasures = new SimilarityMeasure[6];
            similarityMeasures[4]=new StringSim("lev",false);
            //similarityMeasures[5]=new StringSim("lev",false, new SFSquareRoot());
        }
        else
            similarityMeasures = new SimilarityMeasure[4];

		similarityMeasures[0]=new SetOverlap("union",USE_EQUATRE_NORM);
		similarityMeasures[1]=new SetOverlap("union",USE_EQUATRE_NORM, new SFSquareRoot());
		similarityMeasures[2]=new SetOverlap("max",USE_EQUATRE_NORM);
		similarityMeasures[3]=new SetOverlap("max",USE_EQUATRE_NORM, new SFSquareRoot());


		//similarityMeasures[4]=new SetOverlap(2);
		//similarityMeasures[5]=new SetOverlap(2, new SFSquareRoot());
		this.cache=cache;
	}
	@Override
	public Map<Pair<FeatureType, String>, Double> build(String obj1, String obj2) {

		/*if(obj1.contains("leibniz-institute-for-the-social-sciences")&&
				obj2.contains("university-of-koblenz-landau"))
			System.out.println();*/
		log.info("Building features for: "+obj1);
		log.info("\t"+FBOrgName.class.getCanonicalName());
		Pair<FeatureType, List<String>> orgName1 = new FBOrgName(sparqlEndpoint, fn, cache).build(obj1,true);
		log.info("\t"+FBOrgMemberURI.class.getCanonicalName());
		Pair<FeatureType, List<String>> orgMemberURI1 = new FBOrgMemberURI(sparqlEndpoint, cache).build(obj1, true);
		log.info("\t"+FBOrgMemberName.class.getCanonicalName());
		Pair<FeatureType, List<String>> orgMemberName1 = new FBOrgMemberName(sparqlEndpoint,fn, cache).build(obj1,true);
		log.info("\t"+FBOrgParticipatedEventURI.class.getCanonicalName());
		Pair<FeatureType, List<String>> orgParticipatedEventURI1 = new FBOrgParticipatedEventURI(sparqlEndpoint, cache).build(obj1, true);

		log.info("Building features for: "+obj2);
		log.info("\t"+FBOrgName.class.getCanonicalName());
		Pair<FeatureType, List<String>> orgName2 = new FBOrgName(sparqlEndpoint, fn, cache).build(obj2,true);
		log.info("\t"+FBOrgMemberURI.class.getCanonicalName());
		Pair<FeatureType, List<String>> orgMemberURI2 = new FBOrgMemberURI(sparqlEndpoint, cache).build(obj2,true);
		log.info("\t"+FBOrgMemberName.class.getCanonicalName());
		Pair<FeatureType, List<String>> orgMemberName2 = new FBOrgMemberName(sparqlEndpoint,fn, cache).build(obj2,true);
		log.info("\t"+FBOrgParticipatedEventURI.class.getCanonicalName());
		Pair<FeatureType, List<String>> orgParticipatedEventURI2 = new FBOrgParticipatedEventURI(sparqlEndpoint, cache).build(obj2,true);

		Map<Pair<FeatureType, String>, Double> result = new LinkedHashMap<>();
		generateSimilarityMeasures(result, orgName1.getValue(), orgName2.getValue(), orgName1.getKey());
		generateSimilarityMeasures(result, orgMemberName1.getValue(), orgMemberName2.getValue(), orgMemberName1.getKey());
		generateSimilarityMeasures(result, orgMemberURI1.getValue(), orgMemberURI2.getValue(), orgMemberURI1.getKey());
		generateSimilarityMeasures(result, orgParticipatedEventURI1.getValue(), orgParticipatedEventURI2.getValue(), orgParticipatedEventURI1.getKey());

		return result;
	}

	protected void generateSimilarityMeasures(Map<Pair<FeatureType, String>, Double> result,
											  List<String> obj1,
											  List<String> obj2, FeatureType ft){
		for(SimilarityMeasure of : similarityMeasures){
			if(of==null)
				continue;
			double score = of.score(obj1, obj2);
			result.put(new ImmutablePair<>(ft, of.getOption()+"|"+of.getSf()), score);
			if(USE_PRESENCE_FEATURE){
				FeatureGenerator.generatePresenceFeature(result, obj1, obj2, ft);
			}
		}
	}
}