package org.scholarlydata.exp.limes;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;


import java.io.*;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;


/**
 * Created by zqz on 22/02/17.
 */
public class TrainingDataSelector {
    public void generateOrgData(String inputPairCSV, int colId1, int colId2, int colTruth,
                                double trainPercentage, String outFolder) throws IOException {
        Reader inPairCSVFile = new FileReader(inputPairCSV);
        CSVParser parser = CSVFormat.EXCEL.parse(inPairCSVFile);
        List<CSVRecord> allRecords = parser.getRecords();
        allRecords.remove(0);
        Iterator<CSVRecord> it = allRecords.iterator();
        while(it.hasNext()){
            CSVRecord rec = it.next();
            String label = rec.get(colTruth);
            if(label.equals("0"))
                it.remove();
        }

        generateRandomTrainingData(allRecords, colId1, colId2, outFolder,trainPercentage, "org", 1);
        generateRandomTrainingData(allRecords, colId1, colId2, outFolder,trainPercentage,"org",2);
        generateRandomTrainingData(allRecords, colId1, colId2, outFolder,trainPercentage,"org",3);
        generateRandomTrainingData(allRecords, colId1, colId2, outFolder,trainPercentage,"org",4);

    }

    private void generateRandomTrainingData(List<CSVRecord> allRecords, int colId1,
                                            int colId2, String outFolder,
                                            double trainPercentage, String type, int id
                                            ) throws FileNotFoundException {
        Collections.shuffle(allRecords);
        int total = (int) (allRecords.size()*trainPercentage);
        PrintWriter p = new PrintWriter(outFolder+ File.separator+"training_"+type+"_"+id+".nt");
        for(int i=0; i<=total; i++){
            CSVRecord rec = allRecords.get(i);
            p.println(rec.get(colId1)+"\t"+rec.get(colId2));
        }
        p.close();
    }
}
