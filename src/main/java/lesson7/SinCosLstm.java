package lesson7;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedList;
import java.util.List;


/**
 * Created by Joe on 2018/5/16.
 */
public class SinCosLstm {
    public static void main(String[] args) {
        List<Data> data = readFile("");

        RegIterator trainIter = new RegIterator(data, 1, 5, 5);

        // 构建模型
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(1234)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .weightInit(WeightInit.XAVIER)
                .updater(new Nesterovs(0.01, 0.9))
                .list().layer(0, new GravesLSTM.Builder().activation(Activation.TANH).nIn(1).nOut(32)
                        .build())
                .layer(1, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MSE)
                        .activation(Activation.IDENTITY).nIn(32).nOut(1).build())
                .build();

        MultiLayerNetwork network = new MultiLayerNetwork(conf);
        network.setListeners(new ScoreIterationListener(1));
        network.init();

        int epoch = 10;
        for (int i = 0; i < epoch; i++) {
            while (trainIter.hasNext()) {
                DataSet dataSets = trainIter.next();
                network.fit(dataSets);
            }
            trainIter.reset();
        }

    }

    public static List<Data> readFile(String filePath) {
        try (BufferedReader reader = new BufferedReader(new FileReader(filePath))) {
            List<Data> list = new ArrayList<>();

            String tmp;
            while ((tmp = reader.readLine()) != null) {
                String[] splits = tmp.split(",");
                int length = splits.length;

                double[] datas = new double[length - 1];
                for (int i = 0; i < datas.length; i++) {
                    datas[i] = Double.valueOf(splits[i]);
                }

                double label = Double.valueOf(splits[length - 1]);

                list.add(new Data(datas, label));
            }

            return list;
        } catch (IOException e) {
            e.printStackTrace();
        }

        return Collections.emptyList();
    }

    public static List<Double> getPredict(MultiLayerNetwork net, DataSetIterator iterator) {
        List<Double> labels = new LinkedList<>();
        while (iterator.hasNext()) {
            org.nd4j.linalg.dataset.DataSet dataSet = iterator.next();

            INDArray output = net.output(dataSet.getFeatures());

            long[] shape = output.shape();
            for (int i = 0; i < shape[0]; i++) {
                labels.add(output.getDouble(i));
            }
        }
        iterator.reset();

        return labels;
    }
}
