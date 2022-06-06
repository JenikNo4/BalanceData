package cz.simek.balancedatamaven;

import au.com.bytecode.opencsv.CSVParser;
import com.google.common.collect.ImmutableMap;
import com.sgcharts.sparkutil.Smote;
import org.apache.commons.io.IOUtils;
import org.apache.spark.api.java.function.FilterFunction;
import org.apache.spark.mllib.util.MFDataGenerator;
import org.apache.spark.rdd.RDD;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Encoders;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.catalyst.expressions.GenericRowWithSchema;
import org.apache.spark.sql.streaming.DataStreamWriter;
import org.apache.spark.sql.streaming.StreamingQuery;
import org.apache.spark.sql.types.*;
import org.apache.spark.util.random.RandomSampler$;
import scala.collection.Seq;
import weka.core.Instances;
import weka.core.RandomSample;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.supervised.instance.Resample;
import weka.filters.supervised.instance.SMOTE;
import weka.filters.supervised.instance.SpreadSubsample;
import weka.filters.unsupervised.attribute.NumericToNominal;

import java.io.InputStream;
import java.util.*;
import java.util.stream.Collectors;

import static java.rmi.server.LogStream.log;
import static org.apache.spark.sql.functions.*;

public class Main {
    public static Metadata metaFeature = new MetadataBuilder().putString("machine_learning", "FEATURE").putStringArray("custom", new String[]{"test1", "test2"}).build();


    public static void main(String[] args) throws Exception {
//        weka();
        //        Dataset<Row> rowDataset = sparkSession().read().option("inferSchema", "true").csv("src/main/resources/pima-indians-diabetes.csv");
        Dataset<Row> rowDataset = sparkSession().read().option("inferSchema", "false").csv("src/main/resources/glass.csv");
        long allRecords = rowDataset.count();
        String categoryColumnName = "_c9";

        List<Row> categoriesFrequency = rowDataset.groupBy(categoryColumnName).count().orderBy(desc("count")).collectAsList();
        Long max = (long) categoriesFrequency.get(0).get(1);
        System.out.println("MAX: "+max);
        categoriesFrequency.forEach(System.out::println);

        List<Row> allRows = new ArrayList<>();
        categoriesFrequency.forEach(categories -> {
            Dataset<Row> categorySample = rowDataset.filter(col(categoryColumnName).equalTo(categories.get(0)));
            long samples = max - (long) categories.get(1);
            List<Row> allCategoryRows = new ArrayList<>();
            while(samples != 0){
                List<Row> rows1 = categorySample.toJavaRDD().takeSample(false, (int) samples);
                allCategoryRows.addAll(rows1);
                samples = samples-rows1.stream().count();
            }
            allRows.addAll(allCategoryRows);
        });

        Dataset<Row> allRowsDataset = sparkSession().createDataFrame(allRows, rowDataset.schema());
        Dataset<Row> rowDatasetUnion = rowDataset.unionAll(allRowsDataset);
        List<Row> count3 = rowDatasetUnion.groupBy(categoryColumnName).count().orderBy(desc("count")).collectAsList();
        count3.forEach(System.out::println);






        Long count1 = rowDataset.filter(col("_c9").equalTo(1)).count();

        int howManyTake = max.intValue() - count1.intValue();


        Dataset<Row> sample = rowDataset.filter(col("_c9").equalTo(1));
        List<Row> rows1 = sample.toJavaRDD().takeSample(false, howManyTake);


//        Dataset<Row> dataset = sparkSession().createDataFrame(rows1, rowDataset.schema());
//        Dataset<Row> rowDatasetUnion = rowDataset.unionAll(dataset);
//        long count2 = rowDatasetUnion.count();
//        rowDataset.filter(col("_c9").equalTo(1)).count();


        List<Row> rows = rowDataset.groupBy("_c9").count().orderBy("_c9").collectAsList();
        long rowDataCount = rowDataset.count();
        long countCategories = rows.stream().count();

        Map<String, Double> fraction = new HashMap<>();

//        for (Row row : rows)
//        {
//            String o1 = (String) row.get(0);
//            Long o = (Long) row.get(1);
//
////            fraction.put(o1, o.doubleValue()/Long.valueOf(rowDataCount).doubleValue());
//            System.out.println();
//        }
//
        fraction.put("1", 1D);
        fraction.put("2", 1D);
        fraction.put("3", 1D);
        fraction.put("4", 1D);
        fraction.put("5", 2D);
        fraction.put("6", 3D);
        fraction.put("7", 1D);

//        rowDataset.sampleByKey();
        Dataset<Row> sampled = rowDataset.stat().sampleBy("_c9", fraction, 0L);
        List<Row> sampledRows = rowDataset.groupBy("_c9").count().orderBy("_c9").collectAsList();

        List<Row> sampledList = sampled.collectAsList();

        long count = sampled.count();

//        rowDataset.
//
//        DataSource.read();


        Dataset<Row> rowDataset_a = rowDataset.filter(rowDataset.col("_c9").isin(0));
        Dataset<Row> rowDataset_b = rowDataset.filter(rowDataset.col("_c9").isin(1));
        Dataset<Row> minorityClass;
        Dataset<Row> majorityClass;

        long dasetSize = rowDataset.count();

        double overSamplingBalancingRation;
        if (rowDataset_a.count() > rowDataset_b.count()) {

            overSamplingBalancingRation = (double) (dasetSize - rowDataset_b.count()) / dasetSize;
            majorityClass = rowDataset_a;
            minorityClass = rowDataset_b;
        } else {
            overSamplingBalancingRation = (double) (dasetSize - rowDataset_a.count()) / dasetSize;
            majorityClass = rowDataset_b;
            minorityClass = rowDataset_a;
        }

//        double ratio = (double) rowDataset_a.count() / rowDataset_b.count();
//        ratio = Math.ceil(ratio);
        System.out.println(overSamplingBalancingRation);

        double ratio = (double) majorityClass.count() / (double) minorityClass.count();

        Dataset<Row> df_b_oversampled = minorityClass.sample(true, Math.ceil((ratio * 10) / 10));
        Dataset<Row> unionAll = majorityClass.unionAll(df_b_oversampled);
        System.out.println("OversampledSpark");
//        Dataset<Row> c8 = resampleDataset(rowDataset, 0, "_c8", "0", "1");
        System.out.println("Nuly: " + unionAll.filter(rowDataset.col("_c8").isin(0)).count());
        System.out.println("Jednicky: " + unionAll.filter(rowDataset.col("_c8").isin(1)).count());


        long minorityCount = minorityClass.count();
        long minorityCountDistinct = minorityClass.distinct().count();
        long OverSampledCount = df_b_oversampled.count();
        long OverSampledCountDistinct = df_b_oversampled.distinct().count();


        double underSamplingBalancingRation;
        if (rowDataset_a.count() > rowDataset_b.count()) {

            underSamplingBalancingRation = (double) (dasetSize - rowDataset_a.count()) / dasetSize;
            majorityClass = rowDataset_a;
            minorityClass = rowDataset_b;
        } else {
            underSamplingBalancingRation = (double) (dasetSize - rowDataset_b.count()) / dasetSize;
            majorityClass = rowDataset_b;
            minorityClass = rowDataset_a;
        }

        System.out.println(underSamplingBalancingRation);
        Dataset<Row> df_b_undersample = majorityClass.sample(false, 1 / (Math.ceil(ratio * 10) / 10));

        System.out.println("UndersampledSpark");
//        Dataset<Row> c8 = resampleDataset(rowDataset, 0, "_c8", "0", "1");
        System.out.println("Nuly: " + unionAll.filter(rowDataset.col("_c8").isin(0)).count());
        System.out.println("Jednicky: " + unionAll.filter(rowDataset.col("_c8").isin(1)).count());


        long majorityCount = majorityClass.count();
        long majorityCountDistinct = majorityClass.distinct().count();
        long UnderSampledCount = df_b_undersample.count();
        long UnderSampledCountDistinct = df_b_undersample.distinct().count();


        System.out.println("minorityCount: " + minorityCount);
        System.out.println("minorityCountDistinct: " + minorityCountDistinct);
        System.out.println("OverSampledCount: " + OverSampledCount);
        System.out.println("OverSampledCountDistinct: " + OverSampledCountDistinct);

        System.out.println();

        System.out.println("majorityCount: " + majorityCount);
        System.out.println("majorityCountDistinct: " + majorityCountDistinct);
        System.out.println("UnderSampledCount: " + UnderSampledCount);
        System.out.println("UnderSampledCountDistinct: " + UnderSampledCountDistinct);

        System.out.println();


//        DataSource dataSource = new DataSource();

//        DataSource sourceDataset = new DataSource();


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


}
