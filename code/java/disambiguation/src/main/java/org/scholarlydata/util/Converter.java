package org.scholarlydata.util;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVPrinter;
import org.apache.commons.csv.CSVRecord;

import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.Reader;
import java.util.ArrayList;
import java.util.List;

/**
 *
 */
public class Converter {

    public static void main(String[] args) throws IOException {
        String inputRawCSV = args[0];
        String outputCSV=args[1];

        CSVPrinter printer = new CSVPrinter(new FileWriter(outputCSV), CSVFormat.DEFAULT);

        Reader in = new FileReader(inputRawCSV);
        CSVParser parser = CSVFormat.EXCEL.parse(in);
        int count=0;
        List<CSVRecord> records = parser.getRecords();

        for(CSVRecord rec: records){
            if(count==0 || count%2>0){
                List<String> newRec = new ArrayList<>();
                newRec.add(rec.get(0));
                newRec.add(rec.get(1));
                newRec.add(rec.get(2));
                newRec.add(rec.get(rec.size()-1));
                printer.printRecord(newRec);
            }
            count++;
        }
        printer.close();

    }
}
