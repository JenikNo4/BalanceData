package cz.simek.balancedatamaven;

import org.apache.commons.compress.archivers.ar.ArArchiveEntry;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.internal.config.R;
import org.apache.spark.sql.*;
import org.apache.spark.sql.catalyst.expressions.GenericRowWithSchema;
import org.apache.spark.sql.types.*;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.supervised.instance.Resample;
import weka.filters.supervised.instance.SMOTE;
import weka.filters.supervised.instance.SpreadSubsample;
import weka.filters.unsupervised.attribute.NumericToNominal;

import java.util.*;
import java.util.concurrent.atomic.AtomicReference;

import static java.rmi.server.LogStream.log;
import static org.apache.spark.sql.functions.*;

public class Main {
    public static Metadata metaFeature = new MetadataBuilder().putString("machine_learning", "FEATURE").putStringArray("custom", new String[]{"test1", "test2"}).build();


    private static Dataset<Row> oversampleDatasetV2(Dataset<Row> rowDataset, String categoryColumnName) {
        List<Row> categoriesFrequency = rowDataset.groupBy(categoryColumnName).count().orderBy(desc("count")).collectAsList();
        Long max = (long) categoriesFrequency.get(0).get(1);
        System.out.println("MAX: " + max);
        categoriesFrequency.forEach(System.out::println);

        List<Dataset<Row>> finalListOfDataset = new ArrayList<>();
        List<Dataset<Row>> listOfDataset = new ArrayList<>();

        for (Row categories : categoriesFrequency) {
            Dataset<Row> categorySample = rowDataset.filter(col(categoryColumnName).equalTo(categories.get(0)));
            long samples = max % (long) categories.get(1);
            long howManyTimes = max / (long) categories.get(1);
            List<Row> restRows = new ArrayList<>();
            for (int x = 0; x < howManyTimes - 1; x++) {
                listOfDataset.add(categorySample.sample(1D));
//                categorySample = categorySample.unionAll();
            }
            if (samples > 0) {
                restRows = categorySample.toJavaRDD().takeSample(false, (int) samples);
            }

            Dataset<Row> dataFrame = sparkSession().createDataFrame(restRows, rowDataset.schema());
            listOfDataset.add(dataFrame);

        }
        if (finalListOfDataset.isEmpty()) {
            finalListOfDataset.add(listOfDataset.get(0).unionAll(listOfDataset.get(1)));
        }

        for (int y = 2; y < listOfDataset.size(); y++) {
            finalListOfDataset.add(finalListOfDataset.get(y - 2).unionAll(listOfDataset.get(y)));
        }
        Dataset<Row> finalDataset = finalListOfDataset.get(finalListOfDataset.size() - 1);

        return finalDataset.unionAll(rowDataset);
    }


    private static Dataset<Row> oversampleDataset(Dataset<Row> rowDataset, String categoryColumnName) {
        List<Row> categoriesFrequency = rowDataset.groupBy(categoryColumnName).count().orderBy(desc("count")).collectAsList();
        Long max = (long) categoriesFrequency.get(0).get(1);
        System.out.println("MAX: " + max);
        categoriesFrequency.forEach(System.out::println);

        List<Row> allRows = new ArrayList<>();
        categoriesFrequency.forEach(categories -> {
            Dataset<Row> categorySample = rowDataset.filter(col(categoryColumnName).equalTo(categories.get(0)));
            long samples = max - (long) categories.get(1);
            List<Row> allCategoryRows = new ArrayList<>();
            while (samples != 0) {
                List<Row> rows1 = categorySample.toJavaRDD().takeSample(false, (int) samples);
                allCategoryRows.addAll(rows1);
                samples = samples - rows1.stream().count();
            }
            allRows.addAll(allCategoryRows);
        });
        Dataset<Row> allRowsDataset = sparkSession().createDataFrame(allRows, rowDataset.schema());
        Dataset<Row> rowDatasetUnion = rowDataset.unionAll(allRowsDataset);
        return rowDatasetUnion;
    }

    private static Dataset<Row> undersampleDataset(Dataset<Row> rowDataset, String categoryColumnName) {
        List<Row> categoriesFrequency = rowDataset.groupBy(categoryColumnName).count().orderBy(asc("count")).collectAsList();
        Long min = (long) categoriesFrequency.get(0).get(1);
        System.out.println("Min: " + min);
        categoriesFrequency.forEach(System.out::println);

        List<Row> allRows = new ArrayList<>();
        categoriesFrequency.forEach(categories -> {
            Dataset<Row> categorySample = rowDataset.filter(col(categoryColumnName).equalTo(categories.get(0)));
            long samples = min;
            List<Row> allCategoryRows = new ArrayList<>();
            List<Row> rows1 = categorySample.toJavaRDD().takeSample(false, (int) samples);
            allCategoryRows.addAll(rows1);

            allRows.addAll(allCategoryRows);
        });
        Dataset<Row> allRowsDataset = sparkSession().createDataFrame(allRows, rowDataset.schema());
        return allRowsDataset;
    }

    private static Dataset<Row> undersampleDatasetV2(Dataset<Row> rowDataset, String categoryColumnName) {
        List<Row> categoriesFrequency = rowDataset.groupBy(categoryColumnName).count().orderBy(asc("count")).collectAsList();
        Long min = (long) categoriesFrequency.get(0).get(1);
        System.out.println("Min: " + min);
        categoriesFrequency.forEach(System.out::println);

        List<Dataset<Row>> finalListOfDataset = new ArrayList<>();
        List<Dataset<Row>> listOfDataset = new ArrayList<>();

        categoriesFrequency.forEach(categories -> {
            Dataset<Row> categorySample = rowDataset.filter(col(categoryColumnName).equalTo(categories.get(0)));
            Dataset<Row> sampledCategoryDataset = categorySample.sample(min / categorySample.count());
            listOfDataset.add(sampledCategoryDataset);
            long count = sampledCategoryDataset.count();
            if (count < min) {
                List<Row> restRows = categorySample.toJavaRDD().takeSample(false, (int) (min - count));
                Dataset<Row> restRowsDataSet = sparkSession().createDataFrame(restRows, rowDataset.schema());
                listOfDataset.add(restRowsDataSet);
            } else if (count > min) {
                System.out.println();
            }

        });

        finalListOfDataset.add(listOfDataset.get(0).unionAll(listOfDataset.get(1)));

        for (int y = 2; y < listOfDataset.size(); y++) {
            finalListOfDataset.add(finalListOfDataset.get(y - 2).unionAll(listOfDataset.get(y)));
        }
        Dataset<Row> finalDataset = finalListOfDataset.get(finalListOfDataset.size() - 1);

        return finalDataset;
    }

    public static void main(String[] args) throws Exception {
//        weka();
//        Dataset<Row> rowDataset = sparkSession().read().option("inferSchema", "true").csv("src/main/resources/pima-indians-diabetes.csv");

        Dataset<Row> rowDataset = sparkSession().read().option("inferSchema", "false").csv(paths);
        String categoryColumnName = "_c9";
// Compute column summary statistics.
//        MultivariateStatisticalSummary summary = Statistics.colStats(mat.rdd());
//        System.out.println(summary.mean()); // a dense vector containing the mean value for each column
//        System.out.println(summary.variance()); // column-wise variance
//        System.out.println(summary.numNonzeros()); // number of nonzeros in each column

        Dataset<Row> rowDatasetUnionV2 = undersampleDatasetV2(rowDataset, categoryColumnName);
        Dataset<Row> describe1 = rowDatasetUnionV2.describe();
        describe1.show();
        List<Row> count5 = rowDatasetUnionV2.groupBy(categoryColumnName).count().orderBy(desc("count")).collectAsList();

//        Dataset<Row> rowDatasetUnion = oversampleDataset(rowDataset, categoryColumnName);
//        List<Row> count3 = rowDatasetUnion.groupBy(categoryColumnName).count().orderBy(desc("count")).collectAsList();
//
//        Dataset<Row> rowDataset1 = undersampleDataset(rowDataset, categoryColumnName);
//        List<Row> count4 = rowDataset1.groupBy(categoryColumnName).count().orderBy(desc("count")).collectAsList();

//        rowDatasetUnion.foreach(row -> {
//            System.out.println(row);
//        });
//
//        rowDataset1.foreach(row -> {
//            System.out.println(row);
//        });
//        count3.forEach(System.out::println);
//        count4.forEach(System.out::println);
        count5.forEach(System.out::println);

    }

    public static Dataset<Row> generalTestDatasetWithManyDecimals() {
        SparkSession spark = sparkSession();

        StructField f01 = new StructField("f01", DataTypes.StringType, true, metaFeature);
        StructField f02 = new StructField("f02", DataTypes.StringType, true, metaFeature);
        StructType schema = new StructType(new StructField[]{f01, f02});

        Row row1 = new GenericRowWithSchema(new Object[]{"-1", "0"}, schema);
        Row row2 = new GenericRowWithSchema(new Object[]{"-1.", "0."}, schema);
        Row row3 = new GenericRowWithSchema(new Object[]{"-1.501490", "0.797"}, schema);
        Row row4 = new GenericRowWithSchema(new Object[]{"-1.501490389", "0.797980"}, schema);
        Row row5 = new GenericRowWithSchema(new Object[]{"-1.501490389867", "0.797980948"}, schema);
        Row row6 = new GenericRowWithSchema(new Object[]{"-1.501490389867235", "0.797980948857"}, schema);
        Row row7 = new GenericRowWithSchema(new Object[]{"-1.501490389867235824", "0.797980948857494"}, schema);
        Row row8 = new GenericRowWithSchema(new Object[]{"-1.501490389867235824548", "0.79798094885749435894"}, schema);

        List<Row> rows = Arrays.asList(row1, row2, row3, row4, row5, row6, row7, row8);
        return spark.createDataFrame(rows, schema);
    }

    public static SparkSession sparkSession() {
        SparkSession sparkSession = SparkSession.builder().master("local[2]").getOrCreate();
        return sparkSession;
    }

    private static Dataset<Row> resampleDataset(Dataset<Row> base, double ratio, String class_column, String key, String secondKey) {
        Dataset<Row> datasetGroupA = base.filter(base.col(class_column).isin(key));
        Dataset<Row> datasetGroupB = base.filter(base.col(class_column).isin(secondKey));

        long totalCountA = datasetGroupA.count();
        long totalCountB = datasetGroupB.count();
        ratio = totalCountA / totalCountB;

        double fraction = totalCountA * ratio / totalCountB;
        Dataset<Row> sampled = datasetGroupB.sample(false, fraction);
        return sampled.unionAll(datasetGroupA);

    }

    public static void weka() throws Exception {
        System.out.println("hello");
//        DataSource source = new DataSource("pima-indians-diabetes.csv");
        DataSource source = new DataSource("src/main/resources/glass.csv");
        Instances data = source.getDataSet();
        int numAttributes = data.numAttributes() - 1;
        data.setClassIndex(data.numAttributes() - 1);
        data.numAttributes();
        int i = data.numInstances();

//        int sampleSize = (int)((m_SampleSizePercent / 100.0)  * ((1 - m_BiasToUniformClass) * numInstancesPerClass[i] + m_BiasToUniformClass * data.numInstances() / numActualClasses));
        try {
            NumericToNominal convert = new NumericToNominal();
            String[] options = new String[2];
            options[0] = "-R";
            options[1] = "10";
            convert.setOptions(options);
            convert.setInputFormat(data);

            Instances newData = Filter.useFilter(data, convert);

            System.out.println("Original dataset");
            long count = newData.stream().count();
            long group1 = newData.stream().filter(c -> c.value(9) == 0).count();
            long group2 = newData.stream().filter(c -> c.value(9) == 1).count();
            System.out.println("0 [nuly]: " + newData.stream().filter(c -> c.value(9) == 0).count());
            System.out.println("1 [jednicky]: " + newData.stream().filter(c -> c.value(9) == 1).count());
            System.out.println("2 [dvojky]: " + newData.stream().filter(c -> c.value(9) == 2).count());
            System.out.println("3 [trojky]: " + newData.stream().filter(c -> c.value(9) == 3).count());
            System.out.println("4 [ctyrky]: " + newData.stream().filter(c -> c.value(9) == 4).count());
            System.out.println("5 [petky]: " + newData.stream().filter(c -> c.value(9) == 5).count());
            System.out.println("6 [sestky]: " + newData.stream().filter(c -> c.value(9) == 6).count());
            System.out.println("7 [sedmicky]: " + newData.stream().filter(c -> c.value(9) == 7).count());
//
//            System.out.println("Before");
//            for(int z=0; z<8; z=z+1) {
//                System.out.println("Nominal? "+data.attribute(z).isNominal());
//            }
//
//            System.out.println("After");
//            for(int z=0; z<8; z=z+1) {
//                System.out.println("Nominal? "+newData.attribute(z).isNominal());
//            }

            SpreadSubsample spreadSubsample = new SpreadSubsample();
            spreadSubsample.setDistributionSpread(1);
            spreadSubsample.setInputFormat(newData);
            Instances instances = SpreadSubsample.useFilter(newData, spreadSubsample);
            System.out.println("SpreadSubsample");
            System.out.println("0 [nuly]: " + instances.stream().filter(c -> c.value(9) == 0).count());
            System.out.println("1 [jednicky]: " + instances.stream().filter(c -> c.value(9) == 1).count());
            System.out.println("2 [dvojky]: " + instances.stream().filter(c -> c.value(9) == 2).count());
            System.out.println("3 [trojky]: " + instances.stream().filter(c -> c.value(9) == 3).count());
            System.out.println("4 [ctyrky]: " + instances.stream().filter(c -> c.value(9) == 4).count());
            System.out.println("5 [petky]: " + instances.stream().filter(c -> c.value(9) == 5).count());
            System.out.println("6 [sestky]: " + instances.stream().filter(c -> c.value(9) == 6).count());
            System.out.println("7 [sedmicky]: " + instances.stream().filter(c -> c.value(9) == 7).count());


            SMOTE smote = new SMOTE();
            smote.setInputFormat(newData);
            Instances instancesSmote = SMOTE.useFilter(newData, smote);
            System.out.println("SMOTE: ");
            System.out.println("0 [nuly]: " + instancesSmote.stream().filter(c -> c.value(9) == 0).count());
            System.out.println("1 [jednicky]: " + instancesSmote.stream().filter(c -> c.value(9) == 1).count());
            System.out.println("2 [dvojky]: " + instancesSmote.stream().filter(c -> c.value(9) == 2).count());
            System.out.println("3 [trojky]: " + instancesSmote.stream().filter(c -> c.value(9) == 3).count());
            System.out.println("4 [ctyrky]: " + instancesSmote.stream().filter(c -> c.value(9) == 4).count());
            System.out.println("5 [petky]: " + instancesSmote.stream().filter(c -> c.value(9) == 5).count());
            System.out.println("6 [sestky]: " + instancesSmote.stream().filter(c -> c.value(9) == 6).count());
            System.out.println("7 [sedmicky]: " + instancesSmote.stream().filter(c -> c.value(9) == 7).count());


            /////// OverSample
            double percentageOfMajorityClass = 0;
            if (group1 > group2) {
                percentageOfMajorityClass = ((double) group1 / count) * 100;
            } else {
                percentageOfMajorityClass = ((double) group2 / count) * 100;
            }


            Resample resample = new Resample();
            resample.setNoReplacement(false);
            resample.setBiasToUniformClass(1);
            resample.setSampleSizePercent(Math.ceil(percentageOfMajorityClass * 10 * 2) / 10);
            resample.setInputFormat(newData);

            Instances useFilter = Filter.useFilter(newData, resample);
            useFilter.numAttributes();
            useFilter.numInstances();
            System.out.println("Resample OverSample");
            System.out.println("0 [nuly]: " + useFilter.stream().filter(c -> c.value(9) == 0).count());
            System.out.println("1 [jednicky]: " + useFilter.stream().filter(c -> c.value(9) == 1).count());
            System.out.println("2 [dvojky]: " + useFilter.stream().filter(c -> c.value(9) == 2).count());
            System.out.println("3 [trojky]: " + useFilter.stream().filter(c -> c.value(9) == 3).count());
            System.out.println("4 [ctyrky]: " + useFilter.stream().filter(c -> c.value(9) == 4).count());
            System.out.println("5 [petky]: " + useFilter.stream().filter(c -> c.value(9) == 5).count());
            System.out.println("6 [sestky]: " + useFilter.stream().filter(c -> c.value(9) == 6).count());
            System.out.println("7 [sedmicky]: " + useFilter.stream().filter(c -> c.value(9) == 7).count());


        } catch (Exception e) {
            log("Error when resampling input data!");
            e.printStackTrace();
        }


    }

    static String[] paths = new String[]{"src/main/resources/glass.csv"
            , "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv",
            "src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv","src/main/resources/glass.csv"
    };

}
