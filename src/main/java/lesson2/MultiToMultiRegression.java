package lesson2;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.Random;

/**
 * Created by Joe on 2018/5/20.
 * 多元到多元回归
 */
public class MultiToMultiRegression {
    //随机数种子，用于结果复现
    private static final int seed = 12345;
    //对于每个miniBatch的迭代次数
    private static final int iterations = 10;
    //epoch数量(全部数据的训练次数)
    private static final int nEpochs = 2000;

    private static final int numHiddenNodes = 128;
    //一共生成多少样本点
    private static final int nSamples = 1000;
    //Batch size: i.e., each epoch has nSamples/batchSize parameter updates
    private static final int batchSize = 100;
    //网络模型学习率
    private static final double learningRate = 0.01;
    //随机数据生成的范围
    private static int MIN_RANGE = 0;
    private static int MAX_RANGE = 3;

    private static final Random rng = new Random(seed);

    public static void main(String[] args) {
        //Create the network
        int numInput = 2;
        int numOutputs = 2;
        MultiLayerNetwork net = new MultiLayerNetwork(new NeuralNetConfiguration.Builder()
                .seed(seed)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .weightInit(WeightInit.XAVIER)
                .updater(new Sgd(learningRate))
                .list()
                .layer(0, new DenseLayer.Builder().nIn(numInput).nOut(numHiddenNodes)
                        .activation(Activation.LEAKYRELU).build())
                .layer(1, new DenseLayer.Builder().nIn(numHiddenNodes).nOut(numHiddenNodes)
                        .activation(Activation.LEAKYRELU).build())
                .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                        .activation(Activation.IDENTITY)
                        .nIn(numHiddenNodes).nOut(numOutputs).build())
                .pretrain(false).backprop(true).build()
        );
        net.init();
        net.setListeners(new ScoreIterationListener(1));

        for( int i=0; i<nEpochs; i++ ){
            for (int j = 0; j < 50; j++) {
                net.fit(getTrainingData());
            }
        }

        final INDArray input = Nd4j.create(new double[] { 2, 0.3 }, new int[] { 1, 2 });
        INDArray out = net.output(input, false);
        System.out.println(out);
    }

    public static DataSet getTrainingData() {
        float[][] feature = new float[batchSize][2];
        float[][] labels = new float[batchSize][2];

        for (int i = 0; i < batchSize; i++) {
            feature[i][0] = MIN_RANGE + (MAX_RANGE - MIN_RANGE) * rng.nextFloat();
            feature[i][1] = MIN_RANGE + (MAX_RANGE - MIN_RANGE) * rng.nextFloat();

            labels[i][0] = feature[i][0] + feature[i][1];
            labels[i][1] = feature[i][0] * feature[i][1];
        }

        INDArray featureNdarray = Nd4j.create(feature);
        INDArray labelsNdarray = Nd4j.create(labels);

        return new DataSet(featureNdarray, labelsNdarray);
    }
}
