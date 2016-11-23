package org.scholarlydata.exp;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVPrinter;
import org.apache.commons.csv.CSVRecord;
import org.apache.commons.io.FileUtils;
import org.apache.commons.lang3.tuple.Pair;
import org.scholarlydata.feature.FeatureType;
import org.scholarlydata.feature.pair.PairFBOrg;
import org.scholarlydata.feature.pair.PairFBPer;

import java.io.*;
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
        List<String> stopwords = FileUtils.readLines(new File("stopwords.txt"));
        PairFBOrg fborg = new PairFBOrg(sparqlEndpoint);
        PairFBPer fbper = new PairFBPer(sparqlEndpoint, stopwords);

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
            String uri1 = rec.get(1).trim();
            String uri2 = rec.get(2).trim();
            String idx = rec.get(0).trim();
            String truth = rec.get(3).trim();
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
        }
        printer.close();


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

}
