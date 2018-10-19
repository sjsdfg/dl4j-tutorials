package baidudianshi;

import org.datavec.api.io.filters.RandomPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.*;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.graph.PreprocessorVertex;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.preprocessor.CnnToFeedForwardPreProcessor;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.model.VGG16;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.primitives.Pair;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

/**
 * Created by Joe on 2018/10/18.
 */
public class VGG16Transfer {

    protected static final Logger log = LoggerFactory.getLogger(VGG16Transfer.class);

    protected static int height = 256;
    protected static int width = 256;
    protected static int channels = 3;
    protected static int batchSize = 5;
    protected static Random rng = new Random(123);
    protected static double splitTrainTest = 0.9;

    private static int numLabels;

    public static void main(String[] args) throws Exception {

        log.info("开始读取文件");

        // 开始读取文件
        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
        File mainPath = new File("data/train");

        // File input split. Splits up a root directory in to files.
        FileSplit fileSplit = new FileSplit(mainPath, NativeImageLoader.ALLOWED_FORMATS, rng);
        // 获取样本数量
        int numExamples = Long.valueOf(fileSplit.length()).intValue();
        log.info("总共的样本数量为{}", numExamples);

        // 获取类别数量
        // 因为 mainPath 里面只包含训练数据
        // 且训练数据中的每一个 文件夹 名称均为数据的标签，因此对于文件夹的个数就是我们需要训练的标签个数
        numLabels = fileSplit.getRootDir()
                .listFiles(File::isDirectory)
                .length;

        RandomPathFilter randomPathFilter = new RandomPathFilter(rng);
        InputSplit[] inputSplits = fileSplit.sample(randomPathFilter, splitTrainTest, 1 - splitTrainTest);
        InputSplit trainData = inputSplits[0];
        InputSplit testData = inputSplits[1];

        log.info("trainData URI String Length = {}, testData URI String Length = {}", trainData.length(), testData.length());


        log.info("开始数据增强");
        // Crops images deterministically or randomly.
        ImageTransform cropTransform = new CropImageTransform(10);
        // Flips images deterministically or randomly.
        ImageTransform filpTransform = new FlipImageTransform(rng);
        // Rotates and scales images deterministically or randomly
        ImageTransform rotateTransform30 = new RotateImageTransform(30);
        ImageTransform rotateTransform60 = new RotateImageTransform(60);
        ImageTransform rotateTransform90 = new RotateImageTransform(90);
        ImageTransform rotateTransform120 = new RotateImageTransform(120);
        // Warps the perspective of images deterministically or randomly.
        ImageTransform warpTransform = new WarpImageTransform(rng, 42);

        List<Pair<ImageTransform, Double>> pipeline = Arrays.asList(new Pair<>(cropTransform, 0.9),
                new Pair<>(filpTransform, 0.9),
                new Pair<>(rotateTransform30, 0.9),
//                new Pair<>(rotateTransform60, 0.9),
//                new Pair<>(rotateTransform90, 0.9),
//                new Pair<>(rotateTransform120, 0.9),
                new Pair<>(warpTransform, 0.9));

        // {@link org.datavec.image.transform.PipelineImageTransform.doTransform}
        PipelineImageTransform pipelineImageTransform = new PipelineImageTransform(pipeline, true);

        // 数据标准化
        DataNormalization scaler = new ImagePreProcessingScaler(0, 1);

        // 构造数据模型
        ZooModel zooModel = VGG16.builder().build();
        ComputationGraph vgg16 = (ComputationGraph) zooModel.initPretrained();

        FineTuneConfiguration fineTuneConf = new FineTuneConfiguration.Builder()
                .updater(new Nesterovs(0.1, 0.9))
                .seed(123)
                .build();

        log.info(vgg16.summary());

        ComputationGraph vgg16Transfer = new TransferLearning.GraphBuilder(vgg16)
                .fineTuneConfiguration(fineTuneConf)
                .setFeatureExtractor("fc1")
                .removeVertexKeepConnections("flatten")
                .addVertex("flatten",
                        new PreprocessorVertex(new CnnToFeedForwardPreProcessor(8, 8, 512)),
                        "block5_pool")
                .removeVertexKeepConnections("fc1")
                .addLayer("fc1",
                        new DenseLayer.Builder().nIn(512 * 8 * 8).nOut(4096)
                                .weightInit(WeightInit.DISTRIBUTION)
                                .activation(Activation.RELU).build(),
                        "flatten")
                .removeVertexKeepConnections("predictions") //replace the functionality of the final vertex
                .addLayer("predictions",
                        new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                                .nIn(4096).nOut(6)
                                .weightInit(WeightInit.DISTRIBUTION)
                                .dist(new NormalDistribution(0, 0.2 * (2.0 / (4096 + 6)))) //This weight init dist gave better results than Xavier
                                .activation(Activation.SOFTMAX).build(),
                        "fc2")
                .build();

        log.info(vgg16Transfer.summary());

        vgg16Transfer.setListeners(new ScoreIterationListener(100));

        // 构造训练迭代器
        ImageRecordReader recordReader = new ImageRecordReader(height, width, channels, labelMaker);
        recordReader.initialize(trainData);
        DataSetIterator originIterator = new RecordReaderDataSetIterator(recordReader, batchSize, 1, numLabels);

        ImageRecordReader  transformReader = new ImageRecordReader(height, width, channels, labelMaker);
        transformReader.initialize(trainData, pipelineImageTransform);
        DataSetIterator transformIterator = new RecordReaderDataSetIterator(transformReader, batchSize, 1, numLabels);

        // 训练数据标准化信息
        scaler.fit(originIterator);
        originIterator.setPreProcessor(scaler);
        transformIterator.setPreProcessor(scaler);

        for (int i = 0; i < 10; i++) {
            log.info("==================={}===================", i);
            vgg16Transfer.fit(originIterator);
            vgg16Transfer.fit(transformIterator);
        }

        // 测试集
        ImageRecordReader testReader = new ImageRecordReader(height, width, channels, labelMaker);
        testReader.initialize(testData);
        DataSetIterator testIterator = new RecordReaderDataSetIterator(testReader, batchSize, 1, numLabels);
        testIterator.setPreProcessor(scaler);
        Evaluation eval = vgg16Transfer.evaluate(testIterator);

        log.info(eval.stats());

        ModelSerializer.writeModel(vgg16Transfer, new File("model/vgg16Transfer.zip"), true, scaler);
    }
}
