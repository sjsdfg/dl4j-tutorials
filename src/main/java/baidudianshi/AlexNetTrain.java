package baidudianshi;

import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.filters.RandomPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.*;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.earlystopping.EarlyStoppingModelSaver;
import org.deeplearning4j.earlystopping.EarlyStoppingResult;
import org.deeplearning4j.earlystopping.saver.InMemoryModelSaver;
import org.deeplearning4j.earlystopping.saver.LocalFileModelSaver;
import org.deeplearning4j.earlystopping.scorecalc.DataSetLossCalculator;
import org.deeplearning4j.earlystopping.termination.MaxEpochsTerminationCondition;
import org.deeplearning4j.earlystopping.trainer.EarlyStoppingTrainer;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.Distribution;
import org.deeplearning4j.nn.conf.distribution.GaussianDistribution;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.primitives.Pair;
import org.nd4j.linalg.schedule.ScheduleType;
import org.nd4j.linalg.schedule.StepSchedule;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

/**
 * Created by Joe on 2018/10/18.
 * 提交得分 - 0.8300
 */
public class AlexNetTrain {

    protected static final Logger log = LoggerFactory.getLogger(AlexNetTrain.class);

    protected static int height = 256;
    protected static int width = 256;
    protected static int channels = 3;
    protected static int batchSize = 8;

    protected static long seed = 42;
    protected static Random rng = new Random(seed);
    protected static double splitTrainTest = 0.95;

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
        ImageTransform rotateTransform0 = new RotateImageTransform(0);
        ImageTransform rotateTransform30 = new RotateImageTransform(30);
        ImageTransform rotateTransform60 = new RotateImageTransform(60);
        ImageTransform rotateTransform90 = new RotateImageTransform(90);
        ImageTransform rotateTransform120 = new RotateImageTransform(120);
        // Warps the perspective of images deterministically or randomly.
        ImageTransform warpTransform = new WarpImageTransform(rng, 42);

        boolean shuffle = false;
        List<Pair<ImageTransform, Double>> pipeline = Arrays.asList(new Pair<>(cropTransform, 0.9),
                new Pair<>(filpTransform, 0.9),
                new Pair<>(rotateTransform0, 1.0),
                new Pair<>(rotateTransform30, 0.9),
                new Pair<>(rotateTransform90, 0.9),
                new Pair<>(rotateTransform120, 0.9),
                new Pair<>(warpTransform, 0.9));

        // {@link org.datavec.image.transform.PipelineImageTransform.doTransform}
        PipelineImageTransform pipelineImageTransform = new PipelineImageTransform(pipeline, true);

        // 数据标准化
        DataNormalization scaler = new ImagePreProcessingScaler(0, 1);

        // 构造数据模型
        MultiLayerNetwork network = alexnetModel();
        network.setListeners(new ScoreIterationListener(100));

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

        // {@link org.deeplearning4j.earlystopping.scorecalc.ClassificationScoreCalculator}
        // 使用早停法进行模型的训练
        LocalFileModelSaver modelSaver = new LocalFileModelSaver("model/earlystopping");
//        InMemoryModelSaver<MultiLayerNetwork> modelSaver = new InMemoryModelSaver<>();
        EarlyStoppingConfiguration<MultiLayerNetwork> esConfiguration = new EarlyStoppingConfiguration.Builder<MultiLayerNetwork>()
                .epochTerminationConditions(new MaxEpochsTerminationCondition(200)) //Max of 50 epochs
                .evaluateEveryNEpochs(1)
                .scoreCalculator(new DataSetAccuracyLossCalculator(originIterator, transformIterator))
                .modelSaver(modelSaver)
                .build();

        EarlyStoppingTrainer trainer = new EarlyStoppingTrainer(esConfiguration, network.getLayerWiseConfigurations(), transformIterator);
        // 早停法配置好之后开始训练
        EarlyStoppingResult<MultiLayerNetwork> result = trainer.fit();
        network = result.getBestModel();


        // 测试集
        ImageRecordReader testReader = new ImageRecordReader(height, width, channels, labelMaker);
        testReader.initialize(testData);
        DataSetIterator testIterator = new RecordReaderDataSetIterator(testReader, batchSize, 1, numLabels);
        testIterator.setPreProcessor(scaler);
        Evaluation eval = network.evaluate(testIterator);

        log.info(eval.stats());

        ModelSerializer.writeModel(network, new File("model/AlexNet.zip"), true, scaler);

    }

    private static ConvolutionLayer convInit(String name, int in, int out, int[] kernel, int[] stride, int[] pad, double bias) {
        return new ConvolutionLayer.Builder(kernel, stride, pad).name(name).nIn(in).nOut(out).biasInit(bias).build();
    }

    private static  ConvolutionLayer conv3x3(String name, int out, double bias) {
        return new ConvolutionLayer.Builder(new int[]{3,3}, new int[] {1,1}, new int[] {1,1}).name(name).nOut(out).biasInit(bias).build();
    }

    private static  ConvolutionLayer conv5x5(String name, int out, int[] stride, int[] pad, double bias) {
        return new ConvolutionLayer.Builder(new int[]{5,5}, stride, pad).name(name).nOut(out).biasInit(bias).build();
    }

    private static  SubsamplingLayer maxPool(String name,  int[] kernel) {
        return new SubsamplingLayer.Builder(kernel, new int[]{2,2}).name(name).build();
    }

    private static  DenseLayer fullyConnected(String name, int out, double bias, double dropOut, Distribution dist) {
        return new DenseLayer.Builder().name(name).nOut(out).biasInit(bias).dropOut(dropOut).dist(dist).build();
    }

    public static  MultiLayerNetwork alexnetModel() {
        /**
         * AlexNet model interpretation based on the original paper ImageNet Classification with Deep Convolutional Neural Networks
         * and the imagenetExample code referenced.
         * http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
         **/

        double nonZeroBias = 1;
        double dropOut = 0.8;

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .weightInit(WeightInit.DISTRIBUTION)
                .dist(new NormalDistribution(0.0, 0.01))
                .activation(Activation.RELU)
                .updater(new Nesterovs(new StepSchedule(ScheduleType.ITERATION, 0.1, 0.1, 100000), 0.9))
                .biasUpdater(new Nesterovs(new StepSchedule(ScheduleType.ITERATION, 0.2, 0.1, 100000), 0.9))
                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer) // normalize to prevent vanishing or exploding gradients
                //.l2(5 * 1e-4)
                .list()
                .layer(0, convInit("cnn1", channels, 96, new int[]{11, 11}, new int[]{4, 4}, new int[]{3, 3}, 0))
                .layer(1, new LocalResponseNormalization.Builder().name("lrn1").build())
                .layer(2, maxPool("maxpool1", new int[]{3,3}))
                .layer(3, conv5x5("cnn2", 256, new int[] {1,1}, new int[] {2,2}, nonZeroBias))
                .layer(4, new LocalResponseNormalization.Builder().name("lrn2").build())
                .layer(5, maxPool("maxpool2", new int[]{3,3}))
                .layer(6,conv3x3("cnn3", 384, 0))
                .layer(7,conv3x3("cnn4", 384, nonZeroBias))
                .layer(8,conv3x3("cnn5", 256, nonZeroBias))
                .layer(9, maxPool("maxpool3", new int[]{3,3}))
                .layer(10, fullyConnected("ffn1", 4096, nonZeroBias, dropOut, new GaussianDistribution(0, 0.005)))
                .layer(11, fullyConnected("ffn2", 4096, nonZeroBias, dropOut, new GaussianDistribution(0, 0.005)))
                .layer(12, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .name("output")
                        .nOut(numLabels)
                        .activation(Activation.SOFTMAX)
                        .build())
                .backprop(true)
                .pretrain(false)
                .setInputType(InputType.convolutional(height, width, channels))
                .build();

        return new MultiLayerNetwork(conf);

    }
}
