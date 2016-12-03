package org.scholarlydata.exp;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVPrinter;
import org.apache.commons.csv.CSVRecord;

import java.io.*;
import java.util.*;

/**
 * Created by zqz on 02/12/16.
 */
public class OutputConsolidator {

    public static void main(String[] args) throws IOException {
        OutputConsolidator gen = new OutputConsolidator();
        /*gen.consolidate(args[0],
                Integer.valueOf(args[1]), Integer.valueOf(args[2]),
                args[3], args[4]);*/
        gen.consolidateMulti(args[0],
                Integer.valueOf(args[1]), Integer.valueOf(args[2]),
                "/home/zqz/Work/scholarlydata/code/python/classifier",
                "/home/zqz/Work/scholarlydata/data/test/output/person/surname_block"
                );
        System.out.println("end");
    }

    public void consolidateMulti(String intputPairCSV,
                                 int colId1, int colId2,
                                 String predictionFolder,
                                 String outputFolder) throws IOException {
        File pf = new File(predictionFolder);
        for(File pff: pf.listFiles()){
            if(pff.getName().startsWith("prediction-")) {
                String outFile = outputFolder + File.separator + pff.getName();

                consolidate(intputPairCSV, colId1, colId2,
                        pff.toString(), outFile);
            }
        }
    }

    public void consolidate(String inputPairCSV, int colId1, int colId2,
                                                 String predictionCSV, String outputFile) throws IOException {
        Reader inPairCSVFile = new FileReader(inputPairCSV);
        CSVParser parser = CSVFormat.EXCEL.parse(inPairCSVFile);
        Iterator<CSVRecord> inputPairs=parser.iterator();

        Reader inPredictionCSVFile = new FileReader(predictionCSV);
        CSVParser parserPrediction = CSVFormat.EXCEL.parse(inPredictionCSVFile);
        Iterator<CSVRecord> predictions = parserPrediction.iterator();

        CSVPrinter printer = new CSVPrinter(new FileWriter(outputFile), CSVFormat.EXCEL);

        int line=0;
        while(inputPairs.hasNext()){
            line++;
            CSVRecord rec = inputPairs.next();
            if(line==1)
                continue;

            CSVRecord pred = predictions.next();

            String id1 = rec.get(colId1).trim();
            String id2 = rec.get(colId2).trim();
            String p = pred.get(0);
            List<String> result = new ArrayList<>();
            result.add(id1);
            result.add(id2);
            result.add(p);
            printer.printRecord(result);

        }
        printer.close();

    }
}
