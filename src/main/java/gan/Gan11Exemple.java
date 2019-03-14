package gan;

import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.conf.preprocessor.CnnToFeedForwardPreProcessor;
import org.deeplearning4j.nn.conf.preprocessor.FeedForwardToCnnPreProcessor;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.layers.BaseLayer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.RmsProp;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;
import java.util.*;

/**
 * cnn方式对抗效果出来，但是鉴别器和学习器学习能力不算太强，生成得慢，效果不算完美
 * @author hezf
 * @date 19/1/21
 */
public class Gan11Exemple {

    private int width;
    private int height;
    private int channel;

    private static int noiseSize = 100;

    public void init(int width, int height, int channel) {
        this.width = width;
        this.height = height;
        this.channel = channel;
    }

    private ComputationGraph net;
    private void network() {
        RmsProp dRms = new RmsProp(1e-5);
        NeuralNetConfiguration.Builder builder = new NeuralNetConfiguration.Builder();
        builder.setSeed(1234);
        builder.setWeightInit(WeightInit.XAVIER);
        builder.setIUpdater(new RmsProp(1e-5));
        builder.activation(Activation.IDENTITY);

        ComputationGraphConfiguration.GraphBuilder graphBuilder = builder.graphBuilder()
                .addInputs("g-n-input", "d-r-input")
                .setOutputs("d-output");

        graphBuilder.addLayer("g-dense-start",
                new DenseLayer.Builder()
                        .nIn(noiseSize)
                        .nOut(4 * 4 * 512)
                        .build(), "g-n-input");

        graphBuilder.addLayer("g-bn-01",
                new BatchNormalization.Builder()
                        .nIn(512)
                        .nOut(512)
                        .decay(1)
                        .activation(Activation.RELU)
                        .build(),
                new FeedForwardToCnnPreProcessor(4, 4, 512),
                "g-dense-start");
        //stride * (in-1) + filter - 2*pad = 1 * (4-1) + 3 = 7
        graphBuilder.addLayer("g-dec-01",
                new Deconvolution2D.Builder(4, 4)
                        .stride(1, 1)
                        .nIn(512)
                        .nOut(256)
                        .convolutionMode(ConvolutionMode.Strict)
                        .activation(Activation.IDENTITY)
                        .build(), "g-bn-01");

        graphBuilder.addLayer("g-bn-02",
                new BatchNormalization.Builder()
                        .nIn(256)
                        .nOut(256)
                        .decay(1)
                        .activation(Activation.RELU)
                        .build(), "g-dec-01");

        graphBuilder.addLayer("g-dec-02",
                new Deconvolution2D.Builder(2, 2)
                        .stride(2, 2)
                        .nIn(256)
                        .nOut(128)
                        .convolutionMode(ConvolutionMode.Truncate)
                        .activation(Activation.RELU)
                        .build(), "g-bn-02");



        graphBuilder.addLayer("g-dec-03",
                new Deconvolution2D.Builder(2, 2)
                        .stride(2, 2)
                        .nIn(128)
                        .nOut(channel)
                        .convolutionMode(ConvolutionMode.Truncate)
                        .activation(Activation.SIGMOID)
                        .build(), "g-dec-02");


        //鉴别器部分
        graphBuilder.addVertex("d-m-01",
                new MergeVertex(), "g-dec-03", "d-r-input");

        graphBuilder.addLayer("d-cnn-01",
                new ConvolutionLayer.Builder(5, 5)
                        .stride(1, 1)
                        .nIn(channel)
                        .nOut(20)
                        .activation(Activation.IDENTITY)
                        .updater(dRms)
                        .build(), new GanCnnInputPreProcessor(height, width, channel), "d-m-01");

        graphBuilder.addLayer("d-max-01",
                new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build(), "d-cnn-01");

        graphBuilder.addLayer("d-cnn-02",
                new ConvolutionLayer.Builder(5, 5)
                        .stride(1, 1)
                        .nIn(20)
                        .nOut(50)
                        .activation(Activation.IDENTITY)
                        .updater(dRms)
                        .build(), "d-max-01");

        graphBuilder.addLayer("d-max-02",
                new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build(), "d-cnn-02");



        graphBuilder.addLayer("d-dense-01",
                new DenseLayer.Builder()
                        .nIn(4 * 4 * 50)
                        .nOut(500)
                        .activation(Activation.LEAKYRELU)
                        .updater(dRms)
                        .build(),
                new CnnToFeedForwardPreProcessor(4, 4, 50), "d-max-02");


        graphBuilder.addLayer("d-output",
                new OutputLayer.Builder(LossFunctions.LossFunction.XENT)
                        .nIn(500)
                        .nOut(1)
                        .updater(dRms)
                        .activation(Activation.SIGMOID)
                        .build(),
                "d-dense-01");


        ComputationGraph network = new ComputationGraph(graphBuilder.build());
        network.init();

        UIServer uiServer = UIServer.getInstance();
        StatsStorage memoryStatsStorage = new InMemoryStatsStorage();
        uiServer.attach(memoryStatsStorage);
        int listenerFrequency = 1;
        network.setListeners(new StatsListener(memoryStatsStorage, listenerFrequency), new ScoreIterationListener(50));

        // 禁用Workspace
        network.getConfiguration().setInferenceWorkspaceMode(WorkspaceMode.NONE);
        network.getConfiguration().setTrainingWorkspaceMode(WorkspaceMode.NONE);

        net = network;
    }

    private void freeze(double gRate, double dRate) {
        Layer[] layers = net.getLayers();
        for (Layer layer : layers) {
            if (layer instanceof BaseLayer) {
                BaseLayer baseLayer = (BaseLayer) layer;
                String layerName = baseLayer.getConf().getLayer().getLayerName();
                // System.out.println("layerName = " + layerName + ", type = " + type);
                if (layerName.contains("g-")) {
                    // System.out.println(layerName + " = " + 0);
                    net.setLearningRate(layerName, gRate);
                } else if (layerName.contains("d-")) {
                    // System.out.println(layerName + " = " + baseLr);
                    net.setLearningRate(layerName, dRate);
                }
            }
        }
    }

    private static double dRate = 1e-4;
    private static double gRate = 1e-3;
    public void trainGen(INDArray nInput, int n) {
        INDArray EmptyRInput = Nd4j.zeros(1, channel, height, width).addi(-9999);
        INDArray[] features = new INDArray[] { nInput, EmptyRInput };

        INDArray trueLabel = Nd4j.ones(new long[] { 1, 1 });

        INDArray[] labels = new INDArray[] { trueLabel };
        this.freeze(gRate, 0);
        net.fit(features, labels);
    }

    public double genPridict(INDArray nInput) {
        INDArray EmptyRInput = Nd4j.zeros(1, channel, height, width).addi(-9999);

        INDArray[] features = new INDArray[] { nInput, EmptyRInput };
        INDArray[] out = net.output(true, features);
        return out[0].toDoubleVector()[0];
    }


    public void trainRealDisc(INDArray input, int n) {
        INDArray EmptyZInput = Nd4j.zeros(new int[] { 1, noiseSize });
        INDArray[] features = new INDArray[] { EmptyZInput, input };

        INDArray trueLabel = Nd4j.ones(new long[] { 1, 1 });
        INDArray[] labels = new INDArray[] {trueLabel};

        this.freeze(0, dRate);
        net.fit(features, labels);
    }

    public void trainXRealDisc(INDArray input, int n) {
        INDArray EmptyZInput = Nd4j.zeros(new int[] { 1, noiseSize });
        INDArray[] features = new INDArray[] { EmptyZInput, input };

        INDArray falseLabel = Nd4j.zeros(new long[] { 1, 1 });
        INDArray[] labels = new INDArray[] {falseLabel};

        this.freeze(0, dRate);
        net.fit(features, labels);
    }

    public double discTruePridict(INDArray input) {
        INDArray EmptyZInput = Nd4j.zeros(new int[] { 1, noiseSize });
        INDArray[] features = new INDArray[] { EmptyZInput, input };

        INDArray[] ret = net.output(true, features);
        return ret[0].toDoubleVector()[0];
    }


    public void trainBadDisc(INDArray nInput, int n) {
        INDArray EmptyRInput = Nd4j.zeros(1, channel, height, width).addi(-9999);
        INDArray[] features = new INDArray[] { nInput, EmptyRInput };

        INDArray falseLabel = Nd4j.zeros(new long[] { 1, 1 });
        INDArray[] labels = new INDArray[] {falseLabel};

        this.freeze(0, dRate);
        net.fit(features, labels);
    }

    public double discFlasePridict(INDArray nInput) {
        INDArray EmptyRInput = Nd4j.zeros(1, channel, height, width).addi(-9999);
        INDArray[] features = new INDArray[] { nInput, EmptyRInput };

        INDArray[] ret = net.output(true, features);
        return ret[0].toDoubleVector()[0];
    }

    public Map<String, INDArray> netParams() {
        return net.paramTable();
    }

    public Map<String, INDArray> copy(Map<String, INDArray> data) {
        Map<String, INDArray> newData = new HashMap<>();
        Set<String> genKeySet = data.keySet();
        for (Iterator<String> iter = genKeySet.iterator(); iter.hasNext();) {
            String key = iter.next();
            INDArray val = data.get(key);
            INDArray newVal = val.dup();
            newData.put(key, newVal);
        }
        return newData;
    }

    public boolean eq(Map<String, INDArray> param1, Map<String, INDArray> param2, boolean isG) {
        boolean eq = true;
        Set<String> genKeySet = param1.keySet();
        for (Iterator<String> iter = genKeySet.iterator(); iter.hasNext();) {
            String key = iter.next();
            String[] keys = key.split("-");
            boolean comp = false;
            if (isG) {
                if ("g".equals(keys[0])) {
                    comp = true;
                }
            } else {
                if ("d".equals(keys[0])) {
                    comp = true;
                }
            }
            if (comp) {
                INDArray disc = param2.get(key);

                INDArray gen = param1.get(key);
                if (!eq(disc, gen)) {
                    eq = false;
                }
            }
        }
        return eq;
    }

    private boolean eq(INDArray a1, INDArray a2) {
        Number number = a1.eq(a2).minNumber();
        return number.intValue() == 1;
    }

    public void saveModel(String modelName) {
        try {
            ModelSerializer.writeModel(net, modelName, true);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static INDArray pic;
    private static INDArray getRealPic(boolean train) {
        if (pic == null) {
            Random rd = new Random();
            int dd = 7;//rd.nextInt(10);
            String dir = "/myself/mnist/mnist_png/training/" + dd;
            if (!train) {
                dir = "/myself/mnist/mnist_png/testing/" + dd;
            }
            File dirFile = new File(dir);
            File[] files = dirFile.listFiles();
            String path = files[rd.nextInt(files.length)].getAbsolutePath();
            //String path = files[0].getAbsolutePath();
            pic = ImageUtils.load(path);
        }
        return pic;
    }

    private static void run1() {
        Gan11Exemple exemple = new Gan11Exemple();
        exemple.init(28, 28, 1);
        exemple.network();

        INDArray nInput = Nd4j.randn(new int[]{1, noiseSize});

        INDArray realPicInput = getRealPic(true).div(255);
        for (int i = 0; i < 10000; i++) {
            System.out.println();
            Map<String, INDArray> param1 = exemple.copy(exemple.netParams());
            exemple.trainRealDisc(realPicInput, i);
            Map<String, INDArray> param2 = exemple.copy(exemple.netParams());
//            System.out.println("param(g)=" + exemple.eq(param1, param2, true));
            System.out.println("param(d)=" + exemple.eq(param1, param2, false));
            System.out.println("disc true ret=" + exemple.discTruePridict(realPicInput));



            System.out.println("disc false ret(before)=" + exemple.discFlasePridict(nInput));
            exemple.trainBadDisc(nInput, i);

            Map<String, INDArray> param4 = exemple.copy(exemple.netParams());
//            System.out.println("param(g)=" + exemple.eq(param3, param4, true));
            System.out.println("param(d)=" + exemple.eq(param2, param4, false));
            System.out.println("disc false ret(after)=" + exemple.discFlasePridict(nInput));


            System.out.println("gen ret(before)=" + exemple.genPridict(nInput));
            GanCnnInputPreProcessor.save = true;
            exemple.trainGen(nInput, i);
            GanCnnInputPreProcessor.save = false;
            Map<String, INDArray> param3 = exemple.copy(exemple.netParams());
            System.out.println("param(g)=" + exemple.eq(param4, param3, true));
            System.out.println("param(d)=" + exemple.eq(param4, param3, false));
            System.out.println("gen ret(after)=" + exemple.genPridict(nInput));

            System.out.println();
            if (i % 100 == 0) {
                exemple.saveModel("/myself/data/code/ideajava/dl4j-myexemples/model/Model_Gan11." + i + ".zip");
            }
        }
    }

    private INDArray genXRealPic() {
        Random rd = new Random();
        double[] data = new double[28*28];
        for (int i = 0; i < data.length; i++) {
            data[i] = rd.nextInt(255);
        }
        return Nd4j.create(data, new int[] {1, 1, 28, 28});
    }

    public static void run2() {
        Gan11Exemple exemple = new Gan11Exemple();
        exemple.init(28, 28, 1);
        exemple.network();

        for (int i = 0; i < 10000; i++) {
            System.out.println();
            if (i % 2 == 0) {
                INDArray realPicInput = getRealPic(true).div(255);
                exemple.trainRealDisc(realPicInput, i);
                realPicInput = getRealPic(true).div(255);
                System.out.println("disc true ret=" + exemple.discTruePridict(realPicInput));
            } else {
                INDArray realXPicInput = exemple.genXRealPic().div(255);
                if (i % 3 == 0) {
                    exemple.trainXRealDisc(realXPicInput, i);
                    System.out.println("disc xtrue trainret=" + exemple.discTruePridict(realXPicInput));
                } else {
                    System.out.println("disc xtrue ret=" + exemple.discTruePridict(realXPicInput));
                }
            }
            if (i % 5 == 0) {
                INDArray realTestPicInput = getRealPic(false).div(255);
                System.out.println("disc true test ret=" + exemple.discTruePridict(realTestPicInput));
            }
        }
    }

    public static void main(String[] args) {
        run1();
    }
}
