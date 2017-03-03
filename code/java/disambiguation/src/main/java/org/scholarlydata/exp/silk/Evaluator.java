package org.scholarlydata.exp.silk;

import org.apache.commons.collections.CollectionUtils;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVPrinter;
import org.apache.commons.csv.CSVRecord;

import java.io.*;
import java.util.*;

/**
 *
 */
public class Evaluator {

    public static void main(String[] args) throws IOException {
        Evaluator eval = new Evaluator();
        //eval.evaluateOrg();
        eval.evaluatePer();
        System.exit(0);

    }

    protected List<String> readGroundTruth(String inFile, int col1, int col2, int colTruth, boolean pos) throws IOException {
        Reader inPairCSVFile = new FileReader(inFile);
        CSVParser parser = CSVFormat.EXCEL.parse(inPairCSVFile);
        Iterator<CSVRecord> inputPairs = parser.iterator();
        List<String> out = new ArrayList<>();
        int count = 0;
        while (inputPairs.hasNext()) {
            count++;
            if (count == 1) {
                inputPairs.next();
                continue;
            }

            CSVRecord rec = inputPairs.next();
            String truth = rec.get(colTruth);

            if (pos && truth.equals("1"))
                out.add(createOrderedString(rec.get(col1), rec.get(col2)));
            else if (!pos && truth.equals("0"))
                out.add(createOrderedString(rec.get(col1), rec.get(col2)));
        }
        parser.close();
        Collections.sort(out);
        return out;
    }

    protected String createOrderedString(String s1, String s2) {
        s1 = trimAngleBrackets(s1);
        s2 = trimAngleBrackets(s2);
        if (s1.compareTo(s2) < 0)
            return s1 + "," + s2;
        return s2 + "," + s1;
    }

    private String trimAngleBrackets(String s1) {
        if (s1.startsWith("<"))
            s1 = s1.substring(1).trim();
        if (s1.endsWith(">"))
            s1 = s1.substring(0, s1.length() - 1).trim();
        return s1;
    }

    /*following 2 methods are created to adapt to Andrea's data format*/
    ///home/zqz/Work/scholarlydata/data/silk_experiment/org_marked
    protected List<String> readPredictions(String inFile, boolean correctOnly) throws IOException {
        Reader inPairCSVFile = new FileReader(inFile);
        CSVParser parser = CSVFormat.EXCEL.parse(inPairCSVFile);
        Iterator<CSVRecord> inputPairs = parser.iterator();

        List<String> res = new ArrayList<>();

        int count = 0;
        while(inputPairs.hasNext()) {
            CSVRecord rec = inputPairs.next();
            /*if(parts[1].contains("https://w3id.org/scholarlydata/organisation/expert-system")
                    ||parts[2].contains("https://w3id.org/scholarlydata/organisation/expert-system")) {
                if(parts[1].contains("https://w3id.org/scholarlydata/organisation/university-of-cambridge")||
                        parts[2].contains("https://w3id.org/scholarlydata/organisation/university-of-cambridge"))
                    System.out.println();
            }*/
            //double score = Double.valueOf(parts[2].trim());

            count++;
            if (count == 1)
                continue;
            if (rec.size() < 3)
                continue;
            if (correctOnly && rec.get(3).toLowerCase().equals("correct")) {
                String toAdd = createOrderedString(rec.get(0), rec.get(1));
                res.add(toAdd);
            } else if (!correctOnly) {
                String toAdd = createOrderedString(rec.get(0), rec.get(1));
                res.add(toAdd);
            }
        }
        Collections.sort(res);
        return res;
    }

    protected void evaluateOrg() throws IOException {
        String groundtruthTestDataFile = "/home/zqz/Work/scholarlydata/data/public/soa_results/SILK/trainAndTestSplits/ORG/csv/org_4_testing_1quarter.csv";
        String predictionFilePos = "/home/zqz/Work/scholarlydata/data/public/soa_results/SILK/SILK-output/ORG/org4_pos.csv";
        String predictionFileNeg = "/home/zqz/Work/scholarlydata/data/public/soa_results/SILK/SILK-output/ORG/org4_neg.csv";

        //String groundtruthTestDataFile="/home/zqz/Work/scholarlydata/data/limes_experiment/per_4_testing_1quarter.csv";
        //String predictionFile="/home/zqz/Work/scholarlydata/data/silk_experiment/output_per_4_prediction.nt";
        String outCSV = "/home/zqz/Work/scholarlydata/data/public/soa_results/SILK/SILK-eval-results/org4.csv";
        CSVPrinter pr = new CSVPrinter(new FileWriter(outCSV), CSVFormat.DEFAULT);


        //read groundtruth
        List<String> truthPos =
                readGroundTruth(groundtruthTestDataFile,
                        1, 2, 3, true);
        List<String> truthNeg =
                readGroundTruth(groundtruthTestDataFile,
                        1, 2, 3, false);
        Collection<String> truthAll = CollectionUtils.union(truthPos, truthNeg);

        List<String> predPosCorrect =
                readPredictions(predictionFilePos, true); //any pairs in the pos_file with  a'correct' annotation
        predPosCorrect.retainAll(truthAll); //discard any pairs not in the ground truth
        List<String> predPosAll = new ArrayList<>(predPosCorrect); //any pairs in the pos_file with a 'correct' annotation
        predPosAll.addAll(expandPosAll(predictionFileNeg));//plus any pairs in the neg_file with a 'incorrect' annotation
        predPosAll.retainAll(truthAll);

        List<String> predNegCorrect =
                readPredictions(predictionFileNeg, true);
        predNegCorrect.retainAll(truthAll);
        List<String> predNegAll = new ArrayList<>(predNegCorrect);
        predNegAll.addAll(expandNegAll(predictionFilePos));
        predNegAll.retainAll(truthAll);

        double pos_p = predPosCorrect.size() / (double) predPosAll.size();
        double pos_r = predPosCorrect.size() / (double) truthPos.size();
        double pos_f1 = 2 * pos_p * pos_r / (pos_p + pos_r);

        double neg_p = predNegCorrect.size() / (double) predNegAll.size();
        double neg_r = predNegCorrect.size() / (double) truthNeg.size();
        double neg_f1 = 2 * neg_p * neg_r / (neg_p + neg_r);

        Collection<String> predCorrectAll = CollectionUtils.union(predPosCorrect, predNegCorrect);
        Collection<String> predAll = CollectionUtils.union(predPosAll, predNegAll);
        double p = predCorrectAll.size() / (double) predAll.size();
        double r = predCorrectAll.size() / (double) truthAll.size();
        double f1 = 2 * p * r / (p + r);

        List<Double> scores = new ArrayList<>();
        scores.add(pos_p);
        scores.add(pos_r);
        scores.add(pos_f1);
        scores.add(neg_p);
        scores.add(neg_r);
        scores.add(neg_f1);
        scores.add(p);
        scores.add(r);
        scores.add(f1);
        pr.printRecord(scores);

        pr.close();
    }

    //looking into predicted positive pairs, if status is <0, it means the system has predicted it to be negativ pair
    private Collection<? extends String> expandNegAll(String predictionFilePos) throws IOException {
        Reader inPairCSVFile = new FileReader(predictionFilePos);
        CSVParser parser = CSVFormat.EXCEL.parse(inPairCSVFile);
        Iterator<CSVRecord> inputPairs = parser.iterator();

        List<String> res = new ArrayList<>();

        int count=0;
        while(inputPairs.hasNext()) {
            count++;
            CSVRecord rec = inputPairs.next();
            if(rec.size()<3||count==1)
                continue;
            Double status = Double.valueOf(rec.get(2));
            if(status<0)
                res.add(createOrderedString(rec.get(0), rec.get(1)));
        }
        parser.close();
        return res;

    }

    //look into predicted negative pairs. If status>0, then the system in fact predicts that pair to be positive
    private List<String> expandPosAll(String predictionFileNeg) throws IOException {
        Reader inPairCSVFile = new FileReader(predictionFileNeg);
        CSVParser parser = CSVFormat.EXCEL.parse(inPairCSVFile);
        Iterator<CSVRecord> inputPairs = parser.iterator();

        List<String> res = new ArrayList<>();

        int count=0;
        while(inputPairs.hasNext()) {
            count++;
            CSVRecord rec = inputPairs.next();
            if(rec.size()<3||count==1)
                continue;
            Double status =Double.valueOf(rec.get(2));
            if(status>=0)
                res.add(createOrderedString(rec.get(0), rec.get(1)));
        }
        parser.close();
        return res;
    }

    protected void evaluatePer() throws IOException {
        String groundtruthTestDataFile = "/home/zqz/Work/scholarlydata/data/public/soa_results/SILK/trainAndTestSplits" +
                "/PER/csv/per_4_testing_1quarter.csv";
        String predictionFilePos = "/home/zqz/Work/scholarlydata/data/public/soa_results/SILK/SILK-output/PER/" +
                "per4_pos.csv";
        String predictionFileNeg = "/home/zqz/Work/scholarlydata/data/public/soa_results/SILK/SILK-output/PER/" +
                "per4_neg.csv";

        //String groundtruthTestDataFile="/home/zqz/Work/scholarlydata/data/limes_experiment/per_4_testing_1quarter.csv";
        //String predictionFile="/home/zqz/Work/scholarlydata/data/silk_experiment/output_per_4_prediction.nt";
        String outCSV = "/home/zqz/Work/scholarlydata/data/public/soa_results/SILK/SILK-eval-results/per4.csv";
        CSVPrinter pr = new CSVPrinter(new FileWriter(outCSV), CSVFormat.DEFAULT);


        //read groundtruth
        List<String> truthPos =
                readGroundTruth(groundtruthTestDataFile,
                        1, 2, 3, true);
        List<String> truthNeg =
                readGroundTruth(groundtruthTestDataFile,
                        1, 2, 3, false);
        Collection<String> truthAll = CollectionUtils.union(truthPos, truthNeg);

        List<String> predPosCorrect =
                readPredictions(predictionFilePos, true);
        predPosCorrect.retainAll(truthAll);
        List<String> predPosAll = new ArrayList<>(predPosCorrect);
        predPosAll.addAll(expandPosAll(predictionFileNeg));
        predPosAll.retainAll(truthAll);

        List<String> predNegCorrect =
                readPredictions(predictionFileNeg, true);
        predNegCorrect.retainAll(truthAll);
        List<String> predNegAll = new ArrayList<>(predNegCorrect);
        predNegAll.addAll(expandNegAll(predictionFilePos));
        predNegAll.retainAll(truthAll);

        double pos_p = predPosCorrect.size() / (double) predPosAll.size();
        double pos_r = predPosCorrect.size() / (double) truthPos.size();
        double pos_f1 = 2 * pos_p * pos_r / (pos_p + pos_r);

        double neg_p = predNegCorrect.size() / (double) predNegAll.size();
        double neg_r = predNegCorrect.size() / (double) truthNeg.size();
        double neg_f1 = 2 * neg_p * neg_r / (neg_p + neg_r);

        Collection<String> predCorrectAll = CollectionUtils.union(predPosCorrect, predNegCorrect);
        Collection<String> predAll = CollectionUtils.union(predPosAll, predNegAll);
        double p = predCorrectAll.size() / (double) predAll.size();
        double r = predCorrectAll.size() / (double) truthAll.size();
        double f1 = 2 * p * r / (p + r);

        List<Double> scores = new ArrayList<>();
        scores.add(pos_p);
        scores.add(pos_r);
        scores.add(pos_f1);
        scores.add(neg_p);
        scores.add(neg_r);
        scores.add(neg_f1);
        scores.add(p);
        scores.add(r);
        scores.add(f1);
        pr.printRecord(scores);

        pr.close();
    }

}