package org.scholarlydata;

import org.scholarlydata.feature.Predicate;

/**
 *
 */
public class SPARQLQueries {

    public static String getObjectsOf(String sub, String pred) {
        StringBuilder sb = new StringBuilder("select distinct ?o where {\n");
        sb.append("<").append(sub).append(">")
                .append(" ")
                .append("<").append(pred).append(">")
                .append("  ?o .}\n");
        return sb.toString();
    }

    public static String getSubjectsOf(String pred, String obj) {
        StringBuilder sb = new StringBuilder("select distinct ?s where {\n");
        sb.append("?s ").
            append("<").append(pred).append(">")
                .append(" ")
                .append("<").append(obj).append("> .}");
        return sb.toString();
    }

    /**
     * a query that represent a second order path, starting from input, follow first relation to get the subject, then follow
     * the second relation to get the object. only distinct objects are shown
     * @return
     */
    public static String pathSubObj(String obj, String firstPred, String secondPred) {
        StringBuilder sb = new StringBuilder("select distinct ?o where {\n");
        sb.append("?s ").
                append("<").append(firstPred).append(">")
                .append(" ")
                .append("<").append(obj).append("> . \n")
                .append("?s <").append(secondPred).append("> ?o .}");
        return sb.toString();
    }

    /**
     * a query that represent a third order path, starting from input, follow first relation to get the subject, then follow
     * the second relation to get the object, finally follow the third relation to get the desired object. only distinct objects are shown
     * @return
     */
    public static String pathSubObjObj(String obj, String firstPred, String secondPred, String thirdPred) {
        StringBuilder sb = new StringBuilder("select distinct ?o where {\n");
        sb.append("?s ").
                append("<").append(firstPred).append(">")
                .append(" ")
                .append("<").append(obj).append("> . \n")
                .append("?s <").append(secondPred).append("> ?o1 . \n")
                .append("?o1 <").append(thirdPred).append("> ?o .}");
        return sb.toString();
    }

    /**
     * a query that represent a second order path, starting from input, follow first relation to get the object, then follow
     * the second relation to get the desired object. only distinct objects are shown
     * @return
     */
    public static String pathObjObj(String obj, String firstPred, String secondPred) {
        StringBuilder sb = new StringBuilder("select distinct ?o where {\n");
        sb.append("<").append(obj).append("> ")
                .append(firstPred)
                .append(" ")
                .append("?o1 . \n")
                .append("?o1 <").append(secondPred).append("> ?o .}");
        return sb.toString();
    }

    /**
     * a query that represent a third order path, starting from input, follow first relation to get the object, then follow
     * the second relation to get the next object, finally follow the third relation to get the final object. only distinct objects are shown
     * @return
     */
    public static String pathObjObjObj(String obj, String firstPred, String secondPred, String thirdPred) {
        StringBuilder sb = new StringBuilder("select distinct ?o where {\n");
        sb.append("<").append(obj).append("> ")
                .append(firstPred)
                .append(" ")
                .append("?o1 . \n")
                .append("?o1 <").append(secondPred).append("> ?o2 . \n")
                .append("?o2 <").append(thirdPred).append("> ?o .}");
        return sb.toString();
    }

}
