package org.scholarlydata.util;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;

import java.io.FileReader;
import java.io.IOException;
import java.io.Reader;
import java.util.*;

/**
 *
 */
public class ClusterGenerator {

    public static void main(String[] args) throws IOException {
        ClusterGenerator gen = new ClusterGenerator();
        Collection<Set<String>> clusters = gen.createCluster(args[0],
                Integer.valueOf(args[1]), Integer.valueOf(args[2]),
                args[3]);
        System.out.println("end");
    }

    public Collection<Set<String>> createCluster(String inputPairCSV, int colId1, int colId2,
                                                 String predictionCSV) throws IOException {
        Reader inPairCSVFile = new FileReader(inputPairCSV);
        CSVParser parser = CSVFormat.EXCEL.parse(inPairCSVFile);
        Map<String, Set<String>> clusters = new HashMap<>();
        List<CSVRecord> records = parser.getRecords();

        Reader inPredictionCSVFile = new FileReader(predictionCSV);
        CSVParser parserPrediction = CSVFormat.EXCEL.parse(inPredictionCSVFile);
        List<CSVRecord> predictions = parserPrediction.getRecords();
        for (int i = 1; i < records.size(); i++) {
            String label = predictions.get(i - 1).toString();
            if (label.equalsIgnoreCase("1")) {
                CSVRecord rec = records.get(i);
                String id1 = rec.get(colId1).trim();
                String id2 = rec.get(colId2).trim();
                mergeToCluster(id1, id2, clusters);
            }
        }
        return clusters.values();
    }

    public void mergeToCluster(String id1, String id2, Map<String, Set<String>> clusters) {
        Set<String> cluster = clusters.get(id1);
        if (cluster == null) {
            cluster = clusters.get(id2);
            if (cluster == null)
                cluster = new HashSet<>();
        }
        cluster.add(id1);
        cluster.add(id2);
        clusters.put(id1, cluster);
        clusters.put(id2, cluster);
    }
}
