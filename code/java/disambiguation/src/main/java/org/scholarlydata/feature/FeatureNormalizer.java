package org.scholarlydata.feature;

import java.text.Normalizer;
import java.util.regex.Pattern;

/**
 *
 */
public class FeatureNormalizer {
    Pattern pattern = Pattern.compile("\\p{InCombiningDiacriticalMarks}+");
    public String normalize(String in){
        String nfdNormalizedString = Normalizer.normalize(in, Normalizer.Form.NFD);

        String asciiString= pattern.matcher(nfdNormalizedString).replaceAll("");
        return asciiString.replaceAll("[^a-zA-Z0-9]"," ").trim().toLowerCase();
    }
}
