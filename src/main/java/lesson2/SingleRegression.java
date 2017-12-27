package lesson2;

import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.List;
import java.util.Random;

/**
 * Created by Joe on 2017/12/27.
 * 一元线性回归
 * 用于拟合 y = 0.5x + 0.1
 */
public class SingleRegression {
    //随机数种子，用于结果复现
    private static final int seed = 12345;
    //对于每个miniBatch的迭代次数
    private static final int iterations = 10;
    //epoch数量(全部数据的训练次数)
    private static final int nEpochs = 20;
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
        int numInput = 1;
        int numOutputs = 1;

        MultiLayerNetwork net = new MultiLayerNetwork(new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(iterations)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .weightInit(WeightInit.XAVIER)
                .updater(new Sgd(learningRate))
                .list()
                .layer(0, new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                        .activation(Activation.IDENTITY)
                        .nIn(numInput).nOut(numOutputs).build())
                .pretrain(false).backprop(true).build()
        );
        net.init();
        net.setListeners(new ScoreIterationListener(1));

        DataSetIterator iterator = getTrainingData(batchSize, rng);


        // 训练整个数据集nEpochs次
        for( int i=0; i<nEpochs; i++ ){
            iterator.reset();
            net.fit(iterator);
        }
        // 测试两个数字，判断
        final INDArray input = Nd4j.create(new double[] { 10, 100 }, new int[] { 2, 1 });
        INDArray out = net.output(input, false);
        System.out.println(out);
    }

    private static DataSetIterator getTrainingData(int batchSize, Random rand) {
        double [] output = new double[nSamples];
        double [] input = new double[nSamples];
        for (int i= 0; i< nSamples; i++) {
            input[i] = MIN_RANGE + (MAX_RANGE - MIN_RANGE) * rand.nextDouble();

            output[i] = 0.5 * input[i] + 0.1;
        }
        INDArray inputNDArray = Nd4j.create(input, new int[]{nSamples,1});

        INDArray outPut = Nd4j.create(output, new int[]{nSamples, 1});
        DataSet dataSet = new DataSet(inputNDArray, outPut);
        List<DataSet> listDs = dataSet.asList();

        return new ListDataSetIterator(listDs,batchSize);
    }
}
