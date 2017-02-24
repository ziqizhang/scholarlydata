package org.scholarlydata.exp.limes;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVPrinter;
import org.apache.commons.csv.CSVRecord;


import java.io.*;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;


/**
 * Created by zqz on 22/02/17.
 */
public class TrainingDataSelector {

    static final String type="per";
    public static void main(String[] args) throws IOException {
        TrainingDataSelector trainingDataSelector = new TrainingDataSelector();

        /*trainingDataSelector.generateOrgData(
                "/home/zqz/Work/scholarlydata/data/train/training_org(expanded).csv",
                1,2,3,0.75,
                "/home/zqz/Work/scholarlydata/data/limes_experiment/");*/

        trainingDataSelector.generateOrgData(
                "/home/zqz/Work/scholarlydata/data/train/training_per.csv",
                1,2,3,0.75,
                "/home/zqz/Work/scholarlydata/data/limes_experiment/");
    }

    public void generateOrgData(String inputPairCSV, int colId1, int colId2, int colTruth,
                                double trainPercentage, String outFolder) throws IOException {
        Reader inPairCSVFile = new FileReader(inputPairCSV);
        CSVParser parser = CSVFormat.EXCEL.parse(inPairCSVFile);
        List<CSVRecord> allRecords = parser.getRecords();
        allRecords.remove(0);

        generateRandomTrainingData(allRecords, colId1, colId2, colTruth,outFolder,trainPercentage, type, 1);
        generateRandomTrainingData(allRecords, colId1, colId2, colTruth, outFolder,trainPercentage,type,2);
        generateRandomTrainingData(allRecords, colId1, colId2, colTruth,outFolder,trainPercentage,type,3);
        generateRandomTrainingData(allRecords, colId1, colId2, colTruth,outFolder,trainPercentage,type,4);

    }

    private void generateRandomTrainingData(List<CSVRecord> allRecords, int colId1,
                                            int colId2, int colTruth, String outFolder,
                                            double trainPercentage, String type, int id
                                            ) throws IOException {
        Collections.shuffle(allRecords);
        int total = (int) (allRecords.size()*trainPercentage);
        CSVPrinter pTrainSplit = new CSVPrinter(new FileWriter(outFolder+ File.separator+
                type+"_"+id+"_training_3quarters.csv"),
                CSVFormat.DEFAULT);
        CSVPrinter pTestSplit = new CSVPrinter(new FileWriter(outFolder+ File.separator+
                type+"_"+id+"_testing_1quarter.csv"),
                CSVFormat.DEFAULT);
        PrintWriter pTrainPosLimesFormat = new PrintWriter(outFolder+ File.separator+type+"_"+id+"_training_3quarters_pos.nt");
        for(int i=0; i<=total; i++){
            CSVRecord rec = allRecords.get(i);
            if(rec.get(colTruth).equals("1"))
            //p.println("<"+rec.get(colId1)+">\t<"+rec.get(colId2)+">\t \"2.0\" .");
            pTrainPosLimesFormat.println("<"+rec.get(colId1)+">\t<http://www.w3.org/2002/07/owl#sameAs>\t<"+rec.get(colId2)+"> .");

            pTrainSplit.printRecord(rec);
        }

        for(int i=total+1; i<allRecords.size(); i++){
            CSVRecord rec = allRecords.get(i);
            pTestSplit.printRecord(rec);
        }
        pTrainPosLimesFormat.close();
        pTestSplit.close();
        pTestSplit.close();
    }
}
