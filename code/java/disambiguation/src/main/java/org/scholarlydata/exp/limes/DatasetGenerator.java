package org.scholarlydata.exp.limes;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;
import org.apache.commons.io.FileUtils;
import org.apache.commons.lang3.tuple.Pair;
import org.apache.log4j.Logger;
import org.apache.solr.client.solrj.embedded.EmbeddedSolrServer;
import org.scholarlydata.feature.FeatureNormalizer;
import org.scholarlydata.feature.FeatureType;
import org.scholarlydata.feature.Predicate;
import org.scholarlydata.feature.org.FBOrgMemberName;
import org.scholarlydata.feature.org.FBOrgMemberURI;
import org.scholarlydata.feature.org.FBOrgName;
import org.scholarlydata.feature.org.FBOrgParticipatedEventURI;
import org.scholarlydata.feature.per.*;
import org.scholarlydata.util.SolrCache;

import javax.print.attribute.standard.MediaSize;
import java.io.*;
import java.nio.file.Paths;

import java.util.Iterator;
import java.util.List;

/**
 * This class prepares input dataset (source A, source B) that limes need
 */
public class DatasetGenerator {

    private static Logger log = Logger.getLogger(DatasetGenerator.class.getName());
    public static final String NAMESPACE="http://scholarlydata.org/deduplication/";
    private String sparqlEndpoint;
    private FeatureNormalizer fn = new FeatureNormalizer();
    private SolrCache cache;
    private List<String> stopwords;


    public static void main(String[] args) throws IOException {
        String scholarlydataTrainingSet=args[0];
        String outputSourceDataset=args[1];
        String outputTargetDataset=args[2];
        String flag=args[3];

        final String sparqlEndpoint = "http://www.scholarlydata.org/sparql/";
        EmbeddedSolrServer solr = new EmbeddedSolrServer(Paths.get(args[4]), "collection1");
        SolrCache cache = new SolrCache(solr);
        List<String> stopwords = FileUtils.readLines(new File("data/stopwords.txt"));

        DatasetGenerator generator = new DatasetGenerator(sparqlEndpoint,
                cache, stopwords);
        PrintWriter soureWriter = new PrintWriter(outputSourceDataset);
        PrintWriter targetWriter = new PrintWriter(outputTargetDataset);
        if(flag.toLowerCase().startsWith("o"))
            generator.generateOrgData(scholarlydataTrainingSet, 1, 2, soureWriter, targetWriter);
        else
            generator.generatePerData(scholarlydataTrainingSet, 1,2, soureWriter, targetWriter);

        soureWriter.close();
        targetWriter.close();
        System.out.println("finished");
        solr.close();
        System.exit(0);
    }

    public DatasetGenerator(String sparqlEndpoint, SolrCache cache, List<String> stopwords){
        this.sparqlEndpoint=sparqlEndpoint;
        this.cache=cache;
        this.stopwords=stopwords;
    }

    public void generateOrgData(String inputPairCSV, int colId1, int colId2,
                                PrintWriter sourceWriter, PrintWriter targetWriter) throws IOException {
        Reader inPairCSVFile = new FileReader(inputPairCSV);
        CSVParser parser = CSVFormat.EXCEL.parse(inPairCSVFile);
        Iterator<CSVRecord> inputPairs=parser.iterator();

        int line=0;
        while(inputPairs.hasNext()){
            line++;
            CSVRecord rec = inputPairs.next();
            if(line==1)
                continue;

            String uriSource = rec.get(colId1).trim();
            String uriTarget = rec.get(colId2).trim();

            log.info("Building features for: "+uriSource);
            log.info("\t"+FBOrgName.class.getCanonicalName());
            Pair<FeatureType, List<String>> orgName1 = new FBOrgName(sparqlEndpoint, fn, cache).build(uriSource);
            log.info("\t"+FBOrgMemberURI.class.getCanonicalName());
            Pair<FeatureType, List<String>> orgMemberURI1 = new FBOrgMemberURI(sparqlEndpoint, cache).build(uriSource);
            log.info("\t"+FBOrgMemberName.class.getCanonicalName());
            Pair<FeatureType, List<String>> orgMemberName1 = new FBOrgMemberName(sparqlEndpoint,fn, cache).build(uriSource);
            log.info("\t"+FBOrgParticipatedEventURI.class.getCanonicalName());
            Pair<FeatureType, List<String>> orgParticipatedEventURI1 = new FBOrgParticipatedEventURI(sparqlEndpoint, cache).build(uriSource);
            writeToFile(sourceWriter, uriSource, "ORG",orgName1, orgMemberURI1, orgMemberName1, orgParticipatedEventURI1);

            log.info("Building features for: "+uriTarget);
            log.info("\t"+FBOrgName.class.getCanonicalName());
            Pair<FeatureType, List<String>> orgName2 = new FBOrgName(sparqlEndpoint, fn, cache).build(uriTarget);
            log.info("\t"+FBOrgMemberURI.class.getCanonicalName());
            Pair<FeatureType, List<String>> orgMemberURI2 = new FBOrgMemberURI(sparqlEndpoint, cache).build(uriTarget);
            log.info("\t"+FBOrgMemberName.class.getCanonicalName());
            Pair<FeatureType, List<String>> orgMemberName2 = new FBOrgMemberName(sparqlEndpoint,fn, cache).build(uriTarget);
            log.info("\t"+FBOrgParticipatedEventURI.class.getCanonicalName());
            Pair<FeatureType, List<String>> orgParticipatedEventURI2 = new FBOrgParticipatedEventURI(sparqlEndpoint, cache).build(uriTarget);
            writeToFile(targetWriter, uriTarget, "ORG",orgName2, orgMemberURI2, orgMemberName2, orgParticipatedEventURI2);

        }
    }

    public void generatePerData(String inputPairCSV, int colId1, int colId2,
                                PrintWriter sourceWriter, PrintWriter targetWriter) throws IOException {
        Reader inPairCSVFile = new FileReader(inputPairCSV);
        CSVParser parser = CSVFormat.EXCEL.parse(inPairCSVFile);
        Iterator<CSVRecord> inputPairs=parser.iterator();

        int line=0;
        while(inputPairs.hasNext()){
            line++;
            CSVRecord rec = inputPairs.next();
            if(line==1)
                continue;

            String uriSource = rec.get(colId1).trim();
            String uriTarget = rec.get(colId2).trim();

            log.info("Building features for: "+uriSource);
            log.info("\t"+FBPerAffliatedOrgName.class.getCanonicalName());
            Pair<FeatureType, List<String>> perAffOrgName1 = new FBPerAffliatedOrgName(sparqlEndpoint, fn, cache).build(uriSource);
            log.info("\t"+FBPerAffliatedOrgURI.class.getCanonicalName());
            Pair<FeatureType, List<String>> perAffOrgURI1 = new FBPerAffliatedOrgURI(sparqlEndpoint, cache).build(uriSource);
            log.info("\t"+FBPerName.class.getCanonicalName());
            Pair<FeatureType, List<String>> perName1 = new FBPerName(sparqlEndpoint,
                    FeatureType.PERSON_NAME, Predicate.PERSON_name, fn, cache).build(uriSource);
            log.info("\t"+FBPerParticipatedEventURI.class.getCanonicalName());
            Pair<FeatureType, List<String>> perParticipatedEventURI1 = new FBPerParticipatedEventURI(sparqlEndpoint, cache).build(uriSource);
            log.info("\t"+ FBPerPublishedWorkURI.class.getCanonicalName());
            Pair<FeatureType, List<String>> perPublishedWorkURI1 = new FBPerPublishedWorkURI(sparqlEndpoint, cache).build(uriSource);
            log.info("\t"+ FBPerRoleAtEvent.class.getCanonicalName());
            Pair<FeatureType, List<String>> perRoleAtEvent1 = new FBPerRoleAtEvent(sparqlEndpoint, cache).build(uriSource);
            log.info("\t"+FBPerCoAuthorURI.class.getCanonicalName());
            Pair<FeatureType, List<String>> perCoAuthor1=new FBPerCoAuthorURI(sparqlEndpoint, cache).build(uriSource);
            log.info("\t"+FBPerPublishedWorkKAT.class.getCanonicalName());
            Pair<FeatureType, List<String>> perKAT1 = new FBPerPublishedWorkKAT(sparqlEndpoint,fn,stopwords, cache).build(uriSource);
            writeToFile(sourceWriter, uriSource, "PER",perAffOrgName1, perAffOrgURI1, perName1, perParticipatedEventURI1,
                    perPublishedWorkURI1, perRoleAtEvent1, perCoAuthor1, perKAT1);

            log.info("Building features for: "+uriTarget);
            log.info("\t"+FBPerAffliatedOrgName.class.getCanonicalName());
            Pair<FeatureType, List<String>> perAffOrgName2 = new FBPerAffliatedOrgName(sparqlEndpoint, fn, cache).build(uriTarget);
            log.info("\t"+FBPerAffliatedOrgURI.class.getCanonicalName());
            Pair<FeatureType, List<String>> perAffOrgURI2 = new FBPerAffliatedOrgURI(sparqlEndpoint, cache).build(uriTarget);
            log.info("\t"+FBPerName.class.getCanonicalName());
            Pair<FeatureType, List<String>> perName2 = new FBPerName(sparqlEndpoint,
                    FeatureType.PERSON_NAME, Predicate.PERSON_name, fn, cache).build(uriTarget);
            log.info("\t"+FBPerParticipatedEventURI.class.getCanonicalName());
            Pair<FeatureType, List<String>> perParticipatedEventURI2 = new FBPerParticipatedEventURI(sparqlEndpoint, cache).build(uriTarget);
            log.info("\t"+ FBPerPublishedWorkURI.class.getCanonicalName());
            Pair<FeatureType, List<String>> perPublishedWorkURI2 = new FBPerPublishedWorkURI(sparqlEndpoint, cache).build(uriTarget);
            log.info("\t"+ FBPerRoleAtEvent.class.getCanonicalName());
            Pair<FeatureType, List<String>> perRoleAtEvent2 = new FBPerRoleAtEvent(sparqlEndpoint, cache).build(uriTarget);
            log.info("\t"+FBPerCoAuthorURI.class.getCanonicalName());
            Pair<FeatureType, List<String>> perCoAuthor2=new FBPerCoAuthorURI(sparqlEndpoint, cache).build(uriTarget);
            log.info("\t"+FBPerPublishedWorkKAT.class.getCanonicalName());
            Pair<FeatureType, List<String>> perKAT2 = new FBPerPublishedWorkKAT(sparqlEndpoint,fn,stopwords, cache).build(uriTarget);
            writeToFile(targetWriter, uriTarget, "PER",perAffOrgName2, perAffOrgURI2, perName2, perParticipatedEventURI2,
                    perPublishedWorkURI2, perRoleAtEvent2, perCoAuthor2, perKAT2);
        }
    }

    private void writeToFile(PrintWriter writer, String uri, String type, Pair<FeatureType, List<String>>... data) {
        StringBuilder sb= new StringBuilder("<"); //uri isa X
        sb.append(uri).append(">\t<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>\t<").append(NAMESPACE)
                .append(type).append("> .");
        writer.println(sb.toString());
        for(Pair<FeatureType, List<String>> feature: data){
            FeatureType ft = feature.getKey();
            List<String> values = feature.getValue();

            for(String v: values){
                sb = new StringBuilder("<");
                sb.append(uri).append(">\t"); //object
                sb.append("<").append(NAMESPACE).append(ft.getName()).append(">\t");//predicate
                sb.append("\"").append(v).append("\"^^<http://www.w3.org/2001/XMLSchema#string> .");
                writer.println(sb.toString());
            }
        }
    }
}
