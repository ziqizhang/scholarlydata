package org.scholarlydata.exp;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVPrinter;
import org.apache.commons.csv.CSVRecord;

import java.io.*;
import java.util.*;

/**
 * Takes blocking output and generate a csv containing pairs for prediction.
 */
public class PairGenerator {


    public static void main(String[] args) throws IOException {
        Map<String, Set<String>> clusters=readBlocks(args[0]);
        generatePairs(clusters, args[1]);
    }

    protected static Map<String, Set<String>> readBlocks(String csv) throws IOException {
        Reader in = new FileReader(csv);
        CSVParser parser = CSVFormat.EXCEL.parse(in);
        Map<String, Set<String>> clusters = new HashMap<>();
        List<CSVRecord> records = parser.getRecords();
        for(int i=1; i<records.size(); i++){
            CSVRecord cr = records.get(i);
            String cid = cr.get(1);
            String uri = cr.get(0).trim();
            Set<String> cluster = clusters.get(cid);
            if(cluster==null)
                cluster=new HashSet<>();
            cluster.add(uri);
            clusters.put(cid, cluster);
        }
        return clusters;
    }

    protected static void generatePairs(Map<String, Set<String>> clusters, String outCSV) throws IOException {
        CSVPrinter printer = new CSVPrinter(new FileWriter(outCSV), CSVFormat.EXCEL);
        List<String> header = new ArrayList<>();
        header.add("index");
        header.add("uri1");
        header.add("uri2");
        header.add("truth");
        printer.printRecord(header);

        int count=1;
        for(Map.Entry<String, Set<String>> e: clusters.entrySet()){
            Set<String> uris = e.getValue();

            List<String> urisL = new ArrayList<>(uris);

            for(int i=0; i<urisL.size(); i++){
                for(int j=i+1; j<urisL.size(); j++){
                    List<String> row = new ArrayList<>();
                    row.add(String.valueOf(count));
                    row.add(urisL.get(i));
                    row.add(urisL.get(j));
                    printer.printRecord(row);
                    count++;
                }
            }
        }
        printer.close();
    }
}
