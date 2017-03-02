package org.scholarlydata.exp.silk;

import org.apache.commons.collections.CollectionUtils;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;

import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.Reader;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.List;

/**
 * Created by zqz on 02/03/17.
 */
public class Tmp {
    public static void main(String[] args) throws IOException {
        String filegs = "/home/zqz/Work/scholarlydata/data/limes_experiment/per_2_testing_1quarter.csv";
        String filepos = "/home/zqz/Work/scholarlydata/data/silk_experiment_old/per_output/output_per3.csv";
        String fileneg = "/home/zqz/Work/scholarlydata/data/silk_experiment_old/per_output/output_per3_neg.csv";

        /*String filegs = "/home/zqz/Work/scholarlydata/data/limes_experiment/org_3_testing_1quarter.csv";
        String filepos = "/home/zqz/Work/scholarlydata/data/silk_experiment_old/org_output/output_org3.csv";
        String fileneg = "/home/zqz/Work/scholarlydata/data/silk_experiment_old/org_output/output_org_neg3.csv";
*/

        List<String> gs = readGS(filegs);
        List<String> pred = readPred(filepos);
        pred.addAll(readPred(fileneg));

        Collection<String> inter = CollectionUtils.intersection(gs, pred);
        System.out.println(inter.size()/(double)gs.size());
        System.out.println(inter.size()/(double)pred.size());

        PrintWriter p = new PrintWriter("gs.txt");
        for(String l: gs)
            p.println(l);
        p.close();

        p = new PrintWriter("pred.txt");
        for(String pp: pred){
            p.println(pp);
        }
        p.close();

    }

    private static List<String> readGS(String filename) throws IOException {
        Reader inPairCSVFile = new FileReader(filename);
        CSVParser parser = CSVFormat.EXCEL.parse(inPairCSVFile);
        List<CSVRecord> inputPairs = parser.getRecords();

        List<String> content= new ArrayList<>();
        for(CSVRecord r: inputPairs){
            if(r.size()<3)
                continue;
            String s = createOrderedString(r.get(1), r.get(2));
            content.add(s);

        }
        Collections.sort(content);
        return content;
    }

    private static List<String> readPred(String filename) throws IOException {
        Reader inPairCSVFile = new FileReader(filename);
        CSVParser parser = CSVFormat.EXCEL.parse(inPairCSVFile);
        List<CSVRecord> inputPairs = parser.getRecords();

        List<String> content= new ArrayList<>();
        for(CSVRecord r: inputPairs){
            if(r.size()<3)
                continue;
            String s = createOrderedString(r.get(0), r.get(1));
            content.add(s);

        }
        Collections.sort(content);
        return content;
    }

    private static String createOrderedString(String s1, String s2) {
        s1 = trimAngleBrackets(s1);
        s2 = trimAngleBrackets(s2);
        if (s1.compareTo(s2) < 0)
            return s1 + "," + s2;
        return s2 + "," + s1;
    }
    private static String trimAngleBrackets(String s1) {
        if (s1.startsWith("<"))
            s1 = s1.substring(1).trim();
        if (s1.endsWith(">"))
            s1 = s1.substring(0, s1.length() - 1).trim();
        return s1;
    }

}
