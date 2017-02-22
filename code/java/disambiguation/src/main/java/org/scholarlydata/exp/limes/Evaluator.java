package org.scholarlydata.exp.limes;

import org.apache.commons.collections.CollectionUtils;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;

import java.io.FileReader;
import java.io.IOException;
import java.io.Reader;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Iterator;
import java.util.List;

/**
 *
 */
public class Evaluator {

    public static void main(String[] args) {

    }


    public void calculate(List<String> predPos, List<String> predNeg,
                          List<String> truthPos, List<String> truthNeg){
        Collection<String> interPos=CollectionUtils.intersection(predPos, truthPos);
        double p_pos = interPos.size()/(double)predPos.size();
        double r_pos = interPos.size()/(double)truthPos.size();
        double f_pos = 2*p_pos*r_pos/(p_pos+r_pos);

        Collection<String> interNeg=CollectionUtils.intersection(predNeg, truthNeg);
        double p_neg = interNeg.size()/(double)predNeg.size();
        double r_neg = interNeg.size()/(double)truthNeg.size();
        double f_neg = 2*p_neg*r_neg/(p_neg+r_neg);

        double p_total = (interPos.size()+interNeg.size())/(double)(predPos.size()+predNeg.size());

        System.out.println("end");
    }

    protected List<String> readPredictionPos(String inFile){
        return new ArrayList<>();
    }
    protected List<String> createPredictionsNeg(List<String> predPos,
                                                List<String> truthPos,
                                                List<String> truthNeg){
        List<String> all = new ArrayList<>(truthPos);
        all.addAll(truthNeg);

        all.removeAll(predPos);
        return all;
    }

    protected List<String> readGroundTruth(String inFile, int col1, int col2, int colTruth, boolean pos) throws IOException {
        Reader inPairCSVFile = new FileReader(inFile);
        CSVParser parser = CSVFormat.EXCEL.parse(inPairCSVFile);
        Iterator<CSVRecord> inputPairs=parser.iterator();
        List<String> out = new ArrayList<>();
        int count=0;
        while(inputPairs.hasNext()){
            count++;
            if(count==1)
                continue;

            CSVRecord rec = inputPairs.next();
            String truth = rec.get(colTruth);

            if(pos&& truth.equals("1"))
                out.add(createOrderedString(rec.get(col1), rec.get(col2)));
            else if(!pos && truth.equals("0"))
                out.add(createOrderedString(rec.get(col1), rec.get(col2)));
        }
        parser.close();
        return out;
    }

    protected String createOrderedString(String s1, String s2){
        if(s1.compareTo(s2)==0)
            return s1+","+s2;
        return s2+","+s1;
    }
}
