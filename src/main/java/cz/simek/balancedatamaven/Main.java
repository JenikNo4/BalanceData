package cz.simek.balancedatamaven;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.catalyst.expressions.GenericRowWithSchema;
import org.apache.spark.sql.types.*;
import scala.Tuple2;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.supervised.instance.Resample;
import weka.filters.supervised.instance.SMOTE;
import weka.filters.supervised.instance.SpreadSubsample;
import weka.filters.unsupervised.attribute.NumericToNominal;

import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.*;
import java.util.concurrent.atomic.AtomicReference;

import static java.rmi.server.LogStream.log;
import static org.apache.spark.sql.functions.*;

public class Main {
    public static Metadata metaFeature = new MetadataBuilder().putString("machine_learning", "FEATURE").putStringArray("custom", new String[]{"test1", "test2"}).build();

    public static void main(String[] args) throws Exception {
//        weka();
//        Dataset<Row> rowDataset = sparkSession().read().option("inferSchema", "true").csv("src/main/resources/pima-indians-diabetes.csv");
        sparkSession().sparkContext().setLogLevel("WARN");
        Dataset<Row> rowDataset = sparkSession().read().option("inferSchema", "false").csv(paths);
        String categoryColumnName = "_c9";

        DateTimeFormatter dtf = DateTimeFormatter.ofPattern("yyyy/MM/dd HH:mm:ss");
        LocalDateTime now = LocalDateTime.now();
        System.out.println(dtf.format(now));
        Dataset<Row> categoriesFrequency = rowDataset.groupBy(categoryColumnName).count().as("count").orderBy(desc("count"));
        now = LocalDateTime.now();
        System.out.println(dtf.format(now));
        categoriesFrequency.show();
        now = LocalDateTime.now();
        System.out.println(dtf.format(now));
// Compute column summary statistics.
//        MultivariateStatisticalSummary summary = Statistics.colStats(mat.rdd());
//        System.out.println(summary.mean()); // a dense vector containing the mean value for each column
//        System.out.println(summary.variance()); // column-wise variance
//        System.out.println(summary.numNonzeros()); // number of nonzeros in each column


        System.out.println("start oversampling");
        now = LocalDateTime.now();
        System.out.println(dtf.format(now));
//        Dataset<Row> rowDatasetUnionV3 = overSampleDataset(rowDataset, categoryColumnName);
        Dataset<Row> rowDatasetUnionV3 = overSampleDatasetV1(rowDataset, categoryColumnName);
        System.out.println("endOfOversampling");
        now = LocalDateTime.now();
        System.out.println(dtf.format(now));
        System.out.println("count Freq");
        Dataset<Row> count5 = rowDatasetUnionV3.groupBy(categoryColumnName).count().as("count").orderBy(desc("count"));
        count5.show();
        now = LocalDateTime.now();
        System.out.println(dtf.format(now));
        System.out.println("done");

//        Dataset<Row> describe1 = rowDatasetUnionV2.describe();
//        describe1.show();

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


    }



    private static SizedDataset power(Dataset<Row> input, long inputSize, long max) {
        SizedDataset result = new SizedDataset(input, inputSize);

        if (inputSize >= max) {
            return result;
        }

        do {
            result = new SizedDataset(result.getDataset().unionAll(result.getDataset()), result.getSize() * 2);
        }
        while (result.getSize() < max);

        return result;
    }

    /**
     * //original David with no computing - inaccurate
     * @param baseDataset
     * @param categoryColumnName
     * @return
     */
    private static Dataset<Row> overSampleDatasetV2(Dataset<Row> baseDataset, String categoryColumnName) {
        //original David with no computing - inaccurate
        final int CATEGORY_COLUMN_INDEX = 0;
        final String COUNT_FIELD = "count";

        if (baseDataset.isEmpty()) {
            return baseDataset;
        }

        Dataset<Row> categoriesFrequency = baseDataset.groupBy(categoryColumnName).count().as(COUNT_FIELD).orderBy(desc(COUNT_FIELD));
        long max = categoriesFrequency.head().getAs(COUNT_FIELD);
        categoriesFrequency.show();
        // has to be final or effectively final
        final AtomicReference<Dataset<Row>> result = new AtomicReference<>(baseDataset.limit(0)); // start with an empty dataset

        categoriesFrequency.toLocalIterator().forEachRemaining(categories -> {
            Object categoryValue = categories.get(CATEGORY_COLUMN_INDEX);
            long categoryCount = categories.getAs(COUNT_FIELD);

            Dataset<Row> categoryDataset = baseDataset.filter(col(categoryColumnName).equalTo(lit(categoryValue)));

            SizedDataset categoryOverSampled = power(categoryDataset, categoryCount, max);
            Dataset<Row> overSampledDataset = categoryOverSampled.getDataset().sample(1.0D * max / categoryOverSampled.getSize());

            result.set(result.get().unionAll(overSampledDataset));
        });

        return result.get();
    }

    private static class SizedDataset {
        private final Dataset<Row> dataset;
        private final long size;

        SizedDataset(Dataset<Row> dataset, long size) {
            this.dataset = dataset;
            this.size = size;
        }

        public Dataset<Row> getDataset() {
            return dataset;
        }

        public long getSize() {
            return size;
        }
    }

    /**
     * //original David solution with computing adds and removing, accurate
     *input 10_980_000
     * 4:23 minutes
     * 27_152_944 records
     *
     * @param baseDataset
     * @param categoryColumnName
     * @return
     */
    private static Dataset<Row> overSampleDatasetV4(Dataset<Row> baseDataset, String categoryColumnName) {
        //original David solution with computing adds and removing
        final int CATEGORY_COLUMN_INDEX = 0;
        final String COUNT_FIELD = "count";

        if (baseDataset.isEmpty()) {
            return baseDataset;
        }

        Dataset<Row> categoriesFrequency = baseDataset.groupBy(categoryColumnName).count().as(COUNT_FIELD).orderBy(desc(COUNT_FIELD));
        long max = categoriesFrequency.head().getAs(COUNT_FIELD);

        // has to be final or effectively final
        final AtomicReference<Dataset<Row>> result = new AtomicReference<>(baseDataset.limit(0)); // start with an empty dataset

        categoriesFrequency.toLocalIterator().forEachRemaining(categories -> {
            Object categoryValue = categories.get(CATEGORY_COLUMN_INDEX);
            long categoryCount = categories.getAs(COUNT_FIELD);

            Dataset<Row> categoryDataset = baseDataset.filter(col(categoryColumnName).equalTo(lit(categoryValue)));

            SizedDataset categoryOverSampled = power(categoryDataset, categoryCount, max);
            Dataset<Row> overSampledDataset = categoryOverSampled.getDataset().sample(1.0D * max / categoryOverSampled.getSize());
            long count = overSampledDataset.count();
            List<Row> restRows = new ArrayList<>();
            if (count < max) {
                System.out.println("je to mensi");
                long l = max - count;
                restRows = categoryDataset.toJavaRDD().takeSample(false, (int) l);
                Dataset<Row> restRowsDataframe = sparkSession().createDataFrame(restRows, baseDataset.schema());
                result.set(result.get().unionAll(restRowsDataframe));
                result.set(result.get().unionAll(overSampledDataset));
            } else if (max < count) {
                System.out.println("je to vetsi");
                result.set(result.get().unionAll(overSampledDataset.limit((int) max)));
            }else {
                System.out.println("je to stejny");
                result.set(result.get().unionAll(overSampledDataset));
            }
        });

        return result.get();
    }


    /**
     * //My solution with exponencial increasing dataset, then plus dataset, then add remaining rows
     *
     * input 10_980_000
     * 4:08 minutes
     * 27_152_944 records
     *
     *
     * @param baseDataset
     * @param categoryColumnName
     * @return
     */
    private static Dataset<Row> overSampleDatasetV1(Dataset<Row> baseDataset, String categoryColumnName) {
        //My solution with exponencial increasing dataset, then plus dataset, then add remaining rows
        //V1
        Dataset<Row> categoriesFrequency = baseDataset.groupBy(categoryColumnName).count().as("count").orderBy(desc("count"));
        long max = categoriesFrequency.head().getAs("count");
        final AtomicReference<Dataset<Row>> finalDataset = new AtomicReference<>(baseDataset.limit(0)); // start with an empty dataset

        categoriesFrequency.toLocalIterator().forEachRemaining(categories -> {
            Dataset<Row> categoryDataset = baseDataset.filter(col(categoryColumnName).equalTo(categories.get(0)));
            final AtomicReference<Dataset<Row>> finalCategoryDataSet = new AtomicReference<>(categoryDataset);
            long finalCategoryCount = (long) categories.get(1);
            long samples = max % (long) categories.get(1);
            long howManyTimes = max / (long) categories.get(1);

            for (int i = 0; (finalCategoryCount * 2) < max; i++) {
                finalCategoryDataSet.set(finalCategoryDataSet.get().unionAll(finalCategoryDataSet.get()));
                finalCategoryCount = finalCategoryCount * 2;
            }
            for (int i = 0; (finalCategoryCount + (long) categories.get(1)) <= max; i++) {
                finalCategoryDataSet.set(finalCategoryDataSet.get().unionAll(categoryDataset));
                finalCategoryCount = finalCategoryCount + (long) categories.get(1);
            }

            if (samples > 0) {
                finalCategoryDataSet.set(finalCategoryDataSet.get().unionAll(finalCategoryDataSet.get().limit((int) samples)));
            }
//            Dataset<Row> restRowsDataframe = sparkSession().createDataFrame(restRows, baseDataset.schema());
            finalDataset.set(finalDataset.get().unionAll(finalCategoryDataSet.get()));
//            finalDataset.set(finalDataset.get().unionAll(restRowsDataframe));
        });

        return finalDataset.get();
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




    public static SparkSession sparkSession() {
        SparkSession sparkSession = SparkSession.builder().master("local[2]").getOrCreate();
        return sparkSession;
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
