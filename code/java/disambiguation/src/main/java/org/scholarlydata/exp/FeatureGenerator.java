package org.scholarlydata.exp;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVPrinter;
import org.apache.commons.csv.CSVRecord;
import org.apache.commons.io.FileUtils;
import org.apache.commons.lang3.tuple.ImmutablePair;
import org.apache.commons.lang3.tuple.Pair;
import org.apache.solr.client.solrj.embedded.EmbeddedSolrServer;
import org.apache.solr.core.CoreContainer;
import org.scholarlydata.feature.FeatureType;
import org.scholarlydata.feature.pair.PairFBOrg;
import org.scholarlydata.feature.pair.PairFBPer;
import org.scholarlydata.util.SolrCache;

import java.io.*;
import java.nio.file.Paths;
import java.util.*;

/**
 *
 */
public class FeatureGenerator {
    protected static String sparqlEndpoint = "http://www.scholarlydata.org/sparql/";

    public static void main(String[] args) throws IOException {
        String inputCSV = args[0];
        int startRow = Integer.valueOf(args[1]);
        int endRow = Integer.valueOf(args[2]);
        String outputCSV = args[3];
        String type = args[4];
        List<String> stopwords = FileUtils.readLines(new File("data/stopwords.txt"));

        int URI1_index=1;
        int URI2_index=2;

        EmbeddedSolrServer solr = new EmbeddedSolrServer(Paths.get(args[5]), "collection1");
        SolrCache cache = new SolrCache(solr);

        PairFBOrg fborg = new PairFBOrg(sparqlEndpoint, cache);
        PairFBPer fbper = new PairFBPer(sparqlEndpoint, stopwords, cache);

        Iterator<CSVRecord> records = read(inputCSV);
        int index = 0;
        while (index < startRow) {
            records.next();
            index++;
        }

        CSVPrinter printer = new CSVPrinter(new FileWriter(outputCSV), CSVFormat.EXCEL);

        boolean generateHeader = true;
        while (index < endRow && records.hasNext()) {
            CSVRecord rec = records.next();
            String uri1 = rec.get(URI1_index).trim();
            String uri2 = rec.get(URI2_index).trim();
            String idx = rec.get(0).trim();
            String truth = rec.size()>3?rec.get(3).trim():"";
            Map<Pair<FeatureType, String>, Double> features = generateRecord(uri1, uri2, type,
                    fbper, fborg);
            if (generateHeader) {
                List<String> headers = new ArrayList<>();
                headers.add("INDEX");
                headers.add("URI_1");
                headers.add("URI_2");
                for (Pair<FeatureType, String> p : features.keySet()) {
                    String header = p.getKey().getName() + "_" + p.getValue();
                    headers.add(header);
                }
                headers.add("TRUTH");
                printer.printRecord(headers);
                generateHeader = false;
                continue;
            }
            List<String> recordValues = new ArrayList<>();

            recordValues.add(idx);
            recordValues.add(uri1);
            recordValues.add(uri2);
            for (Double d : features.values())
                recordValues.add(d.toString());
            recordValues.add(truth);
            printer.printRecord(recordValues);

            index++;
            System.out.println(index+"\t"+uri1+"|"+uri2+" "+new Date());
        }
        printer.close();
        solr.close();
        System.exit(0);

    }

    protected static Map<Pair<FeatureType, String>, Double>
    generateRecord(String obj1, String obj2, String type, PairFBPer fbper,
                   PairFBOrg fborg) {
        if (type.equalsIgnoreCase("org")) {

            Map<Pair<FeatureType, String>, Double> features = fborg.build(obj1, obj2);
            return features;
        } else {

            Map<Pair<FeatureType, String>, Double> features = fbper.build(obj1, obj2);
            return features;
        }
    }

    protected static Iterator<CSVRecord> read(String inFile) throws IOException {
        Reader in = new FileReader(inFile);
        CSVParser parser = CSVFormat.EXCEL.parse(in);
        return parser.iterator();
    }

    public static void generatePresenceFeature(Map<Pair<FeatureType, String>, Double> result,
                                                  List<String> obj1,
                                                  List<String> obj2, FeatureType ft){
        int indicator=0;
        if(obj1.size()!=0)
            indicator++;
        if(obj2.size()!=0)
            indicator++;
        result.put(new ImmutablePair<>(ft, "presence"), (double)indicator);
    }

}
