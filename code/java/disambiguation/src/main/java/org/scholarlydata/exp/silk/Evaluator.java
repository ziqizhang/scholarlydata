package org.scholarlydata.exp.silk;

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
        //eval.evaluateOrg();
        eval.evaluatePer();
        System.exit(0);

        String groundtruthTestDataFile = "/home/zqz/Work/scholarlydata/data/limes_experiment/org_4_testing_1quarter.csv";
        String predictionFile = "/home/zqz/Work/scholarlydata/data/silk_experiment/output_org_4_prediction.nt";

        //String groundtruthTestDataFile="/home/zqz/Work/scholarlydata/data/limes_experiment/per_4_testing_1quarter.csv";
        //String predictionFile="/home/zqz/Work/scholarlydata/data/silk_experiment/output_per_4_prediction.nt";
        String outCSV = "/home/zqz/Work/scholarlydata/data/silk_experiment/summary.csv";
        CSVPrinter p = new CSVPrinter(new FileWriter(outCSV), CSVFormat.DEFAULT);

        for (double threshold = 1.0; threshold > 0.0; threshold = threshold - 0.1) {
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
            scores.add(0, threshold);
            p.printRecord(scores);
        }
        p.close();
    }


    public List<Double> calculate(List<String> predPos, List<String> predNeg,
                                  List<String> truthPos, List<String> truthNeg) {
        List<Double> scores = new ArrayList<>();
        Collection<String> interPos = CollectionUtils.intersection(predPos, truthPos);
        double p_pos = predPos.size() == 0 ? 0 : interPos.size() / (double) predPos.size();
        double r_pos = interPos.size() / (double) truthPos.size();
        double f_pos = (p_pos == 0 || r_pos == 0) ? 0 : 2 * p_pos * r_pos / (p_pos + r_pos);
        scores.add(p_pos);
        scores.add(r_pos);
        scores.add(f_pos);

        Collection<String> interNeg = CollectionUtils.intersection(predNeg, truthNeg);
        double p_neg = predNeg.size() == 0 ? 0 : interNeg.size() / (double) predNeg.size();
        double r_neg = interNeg.size() / (double) truthNeg.size();
        double f_neg = (p_neg == 0 || r_neg == 0) ? 0 : 2 * p_neg * r_neg / (p_neg + r_neg);
        scores.add(p_neg);
        scores.add(r_neg);
        scores.add(f_neg);

        double p_total = (interPos.size() + interNeg.size()) / (double) (predPos.size() + predNeg.size());

        scores.add(p_total);
        scores.add(p_total);
        scores.add(p_total);
        return scores;
    }

    protected List<String> readPredictionPos(String inFile, double threshold) throws IOException {
        List<String> lines = FileUtils.readLines(new File(inFile), Charset.forName("utf8"));
        List<String> res = new ArrayList<>();

        for (String l : lines) {
            String[] parts = l.split("\\s+");
            /*if(parts[1].contains("https://w3id.org/scholarlydata/organisation/expert-system")
                    ||parts[2].contains("https://w3id.org/scholarlydata/organisation/expert-system")) {
                if(parts[1].contains("https://w3id.org/scholarlydata/organisation/university-of-cambridge")||
                        parts[2].contains("https://w3id.org/scholarlydata/organisation/university-of-cambridge"))
                    System.out.println();
            }*/
            //double score = Double.valueOf(parts[2].trim());
            double score = 1.0;
            if (score < threshold)
                continue;
            String toAdd = createOrderedString(parts[0], parts[2]);
            res.add(toAdd);
        }
        return res;
    }

    protected List<String> createPredictionsNeg(List<String> predPos,
                                                List<String> truthPos,
                                                List<String> truthNeg) {
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
    protected List<String> readPredictionsFormatA(String inFile, boolean correctOnly) throws IOException {
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
        String groundtruthTestDataFile = "/home/zqz/Work/scholarlydata/data/limes_experiment/org_4_testing_1quarter.csv";
        String predictionFilePos = "/home/zqz/Work/scholarlydata/data/silk_experiment_old/org_output/output_org4.csv";
        String predictionFileNeg = "/home/zqz/Work/scholarlydata/data/silk_experiment_old/org_output/output_org_neg4.csv";

        //String groundtruthTestDataFile="/home/zqz/Work/scholarlydata/data/limes_experiment/per_4_testing_1quarter.csv";
        //String predictionFile="/home/zqz/Work/scholarlydata/data/silk_experiment/output_per_4_prediction.nt";
        String outCSV = "/home/zqz/Work/scholarlydata/data/silk_experiment_old/org_summary.csv";
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
                readPredictionsFormatA(predictionFilePos, true); //any pairs in the pos_file with  a'correct' annotation
        predPosCorrect.retainAll(truthAll); //discard any pairs not in the ground truth
        List<String> predPosAll = new ArrayList<>(predPosCorrect); //any pairs in the pos_file with a 'correct' annotation
        predPosAll.addAll(expandPosAll(predictionFileNeg));//plus any pairs in the neg_file with a 'incorrect' annotation
        predPosAll.retainAll(truthAll);

        List<String> predNegCorrect =
                readPredictionsFormatA(predictionFileNeg, true);
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

    private Collection<? extends String> expandNegAll(String predictionFilePos) throws IOException {
        Reader inPairCSVFile = new FileReader(predictionFilePos);
        CSVParser parser = CSVFormat.EXCEL.parse(inPairCSVFile);
        Iterator<CSVRecord> inputPairs = parser.iterator();

        List<String> res = new ArrayList<>();

        while(inputPairs.hasNext()) {
            CSVRecord rec = inputPairs.next();
            if(rec.size()<3)
                continue;
            if(rec.get(3).equals("incorrect"))
                res.add(createOrderedString(rec.get(0), rec.get(1)));
        }
        parser.close();
        return res;

    }

    private List<String> expandPosAll(String predictionFileNeg) throws IOException {
        Reader inPairCSVFile = new FileReader(predictionFileNeg);
        CSVParser parser = CSVFormat.EXCEL.parse(inPairCSVFile);
        Iterator<CSVRecord> inputPairs = parser.iterator();

        List<String> res = new ArrayList<>();

        while(inputPairs.hasNext()) {
            CSVRecord rec = inputPairs.next();
            if(rec.size()<3)
                continue;;
            if(rec.get(3).equals("incorrect"))
                res.add(createOrderedString(rec.get(0), rec.get(1)));
        }
        parser.close();
        return res;
    }

    protected void evaluatePer() throws IOException {
        String groundtruthTestDataFile = "/home/zqz/Work/scholarlydata/data/limes_experiment/per_1_testing_1quarter.csv";
        String predictionFilePos = "/home/zqz/Work/scholarlydata/data/silk_experiment_old/per_output/output_per1.csv";
        String predictionFileNeg = "/home/zqz/Work/scholarlydata/data/silk_experiment_old/per_output/output_per1_neg.csv";

        //String groundtruthTestDataFile="/home/zqz/Work/scholarlydata/data/limes_experiment/per_4_testing_1quarter.csv";
        //String predictionFile="/home/zqz/Work/scholarlydata/data/silk_experiment/output_per_4_prediction.nt";
        String outCSV = "/home/zqz/Work/scholarlydata/data/silk_experiment_old/per_summary.csv";
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
                readPredictionsFormatA(predictionFilePos, true);
        predPosCorrect.retainAll(truthAll);
        List<String> predPosAll = new ArrayList<>(predPosCorrect);
        predPosAll.addAll(expandPosAll(predictionFileNeg));
        predPosAll.retainAll(truthAll);

        List<String> predNegCorrect =
                readPredictionsFormatA(predictionFileNeg, true);
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


    protected void evaluatePer2() throws IOException {
        String groundtruthTestDataFile = "/home/zqz/Work/scholarlydata/data/limes_experiment/per_1_testing_1quarter.csv";
        String predictionFilePos = "/home/zqz/Work/scholarlydata/data/silk_experiment_old/per_output/output_per1.csv";
        String predictionFileNeg = "/home/zqz/Work/scholarlydata/data/silk_experiment_old/per_output/output_per1_neg.csv";

        //String groundtruthTestDataFile="/home/zqz/Work/scholarlydata/data/limes_experiment/per_4_testing_1quarter.csv";
        //String predictionFile="/home/zqz/Work/scholarlydata/data/silk_experiment/output_per_4_prediction.nt";
        String outCSV = "/home/zqz/Work/scholarlydata/data/silk_experiment_old/per_summary.csv";
        CSVPrinter pr = new CSVPrinter(new FileWriter(outCSV), CSVFormat.DEFAULT);


        //read groundtruth
        List<String> truthPos =
                readGroundTruth(groundtruthTestDataFile,
                        1, 2, 3, true);
        List<String> truthNeg =
                readGroundTruth(groundtruthTestDataFile,
                        1, 2, 3, false);
        List<String> truthAll = new ArrayList<>(CollectionUtils.union(truthPos, truthNeg));
        Collections.sort(truthAll);

        List<String> predPosCorrect =
                readPredictionsFormatA(predictionFilePos, true);
        predPosCorrect.retainAll(truthAll);
        List<String> predPosAll = new ArrayList<>(predPosCorrect);
        //predPosAll.addAll(expandPosAll(predictionFileNeg));
        predPosAll.retainAll(truthAll);

        List<String> predNegCorrect =
                readPredictionsFormatA(predictionFileNeg, true);
        predNegCorrect.retainAll(truthAll);
        List<String> predNegAll = new ArrayList<>(predNegCorrect);
        //predNegAll.addAll(expandNegAll(predictionFilePos));
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