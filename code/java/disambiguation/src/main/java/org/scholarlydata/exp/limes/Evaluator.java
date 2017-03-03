package org.scholarlydata.exp.limes;

import org.apache.commons.collections.CollectionUtils;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVPrinter;
import org.apache.commons.csv.CSVRecord;
import org.apache.commons.io.FileUtils;

import java.io.*;
import java.nio.charset.Charset;
import java.util.*;

/**
 *
 */
public class Evaluator {

    public static void main(String[] args) throws IOException {
        Evaluator eval = new Evaluator();
        //String groundtruthTestDataFile="/home/zqz/Work/scholarlydata/data/limes_experiment/org_4_testing_1quarter.csv";
        //String predictionFile="/home/zqz/Work/scholarlydata/data/limes_experiment/output_org_prediction.nt";

        /*String groundtruthTestDataFile="/home/zqz/Work/scholarlydata/data/public/soa_results/LIMES/trainAndTestSplits/" +
                "org_1_testing_1quarter.csv";
        String predictionFile="/home/zqz/Work/scholarlydata/data/public/soa_results/LIMES/output/wombat_complete" +
                "/org_1_prediction.nt";
        String outCSV="/home/zqz/Work/scholarlydata/data/public/soa_results/LIMES/output/summary.csv";
        */
        String groundtruthTestDataFile="/home/zqz/Work/scholarlydata/data/public/soa_results/LIMES/trainAndTestSplits/" +
                "per_1_testing_1quarter.csv";
        String predictionFile="/home/zqz/Work/scholarlydata/data/public/soa_results/LIMES/output/wombat_complete" +
                "/per_1_prediction.nt";
        String outCSV="/home/zqz/Work/scholarlydata/data/public/soa_results/LIMES/output/summary.csv";


        CSVPrinter p = new CSVPrinter(new FileWriter(outCSV), CSVFormat.DEFAULT);

        for(double threshold=1.0; threshold>0.0; threshold=threshold-0.1) {
            //read groundtruth
            List<String> truthPos =
                    eval.readGroundTruth(groundtruthTestDataFile,
                            1, 2, 3, true);
            List<String> truthNeg =
                    eval.readGroundTruth(groundtruthTestDataFile,
                            1, 2, 3, false);
            //read predictions (limes only predict positive pairs
            List<String> predPos = eval.
                    readPredictionPos(predictionFile, threshold);
            //we should only keep predPos that are also of interest to our dataset.
            List<String> all = new ArrayList<>(truthPos);
            all.addAll(truthNeg);
            Collections.sort(all);
        /*for(String a: all)
            System.out.println(a);*/
            Collections.sort(predPos);
        /*for(String p: predPos)
            System.out.println(p);*/
            Collection<String> inter = CollectionUtils.intersection(all, predPos);
            predPos.clear();
            predPos.addAll(inter);

            List<String> predNeg = eval.createPredictionsNeg(predPos,
                    truthPos, truthNeg);

            List<Double> scores = eval.calculate(predPos, predNeg, truthPos, truthNeg);
            scores.add(0,threshold);
            p.printRecord(scores);
        }
        p.close();
    }


    public List<Double> calculate(List<String> predPos, List<String> predNeg,
                          List<String> truthPos, List<String> truthNeg){
        List<Double> scores = new ArrayList<>();
        Collection<String> interPos=CollectionUtils.intersection(predPos, truthPos);
        double p_pos = predPos.size()==0?0:interPos.size()/(double)predPos.size();
        double r_pos = interPos.size()/(double)truthPos.size();
        double f_pos = (p_pos==0||r_pos==0)?0:2*p_pos*r_pos/(p_pos+r_pos);
        scores.add(p_pos);
        scores.add(r_pos);
        scores.add(f_pos);

        Collection<String> interNeg=CollectionUtils.intersection(predNeg, truthNeg);
        double p_neg = predNeg.size()==0?0:interNeg.size()/(double)predNeg.size();
        double r_neg = interNeg.size()/(double)truthNeg.size();
        double f_neg = (p_neg==0||r_neg==0)?0:2*p_neg*r_neg/(p_neg+r_neg);
        scores.add(p_neg);
        scores.add(r_neg);
        scores.add(f_neg);

        double p_total = (interPos.size()+interNeg.size())/(double)(predPos.size()+predNeg.size());

        scores.add(p_total);
        scores.add(p_total);
        scores.add(p_total);
        return scores;
    }

    protected List<String> readPredictionPos(String inFile, double threshold) throws IOException {
        List<String> lines = FileUtils.readLines(new File(inFile), Charset.forName("utf8"));
        List<String> res = new ArrayList<>();

        for(String l: lines){
            String[] parts = l.split("\t+");
            if(parts[1].contains("https://w3id.org/scholarlydata/organisation/expert-system")
                    ||parts[2].contains("https://w3id.org/scholarlydata/organisation/expert-system")) {
                if(parts[1].contains("https://w3id.org/scholarlydata/organisation/university-of-cambridge")||
                        parts[2].contains("https://w3id.org/scholarlydata/organisation/university-of-cambridge"))
                System.out.println();
            }
            double score = Double.valueOf(parts[2].trim());
            if(score<threshold)
                continue;
            String toAdd = createOrderedString(parts[0], parts[1]);
            res.add(toAdd);
        }
        return res;
    }
    protected List<String> createPredictionsNeg(List<String> predPos,
                                                List<String> truthPos,
                                                List<String> truthNeg){
        List<String> all = new ArrayList<>(truthPos);
        all.addAll(truthNeg);
        Collections.sort(all);
        Collections.sort(predPos);
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
            if(count==1) {
                inputPairs.next();
                continue;
            }

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
        s1=trimAngleBrackets(s1);
        s2=trimAngleBrackets(s2);
        if(s1.compareTo(s2)<0)
            return s1+","+s2;
        return s2+","+s1;
    }

    private String trimAngleBrackets(String s1) {
        if(s1.startsWith("<"))
            s1=s1.substring(1).trim();
        if(s1.endsWith(">"))
            s1=s1.substring(0, s1.length()-1).trim();
        return s1;
    }
}
