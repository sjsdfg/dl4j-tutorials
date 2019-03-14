package gan;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.AdaGrad;
import org.nd4j.linalg.learning.config.RmsProp;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.*;

/**
 * 全dense模型，对单通道minist小图较有效
 * @author hezf
 * @date 19/1/12
 */
public class Gan4Exemple {

    private static int width;
    private static int height;
    private static int channel = 1;

    private static double dfrate = 1e-2;
    private static double grate = 1e-2;
    private static double drate = 1e-2;

    public static void main(String[] args) throws IOException {
        //int type = ImageIO.read(new File("/myself/tmp/dl4j/gan/data/train/1/1.jpg")).getType();
        //INDArray indArray0 = toINDArrayBGR(ImageIO.read(new File("/myself/tmp/dl4j/gan/data/train/0/0.jpg"))).div(255);
        /**
        INDArray indArray1 = toINDArrayBGR(ImageIO.read(new File("/myself/tmp/dl4j/gan/data/train/1/1.jpg"))).div(255);
        BufferedImage bufferedImage = toBufferedImage(indArray1.mul(255));
        try {
            ImageIO.write(bufferedImage, "jpg", new File("/myself/tmp/dl4j/gan/data/train/1/11.jpg"));

        } catch (IOException e) {
            e.printStackTrace();
        }
         */

        File file = new File("/myself/tmp/dl4j/gan/data/train/0/0.jpg");
        if (file.exists()) {
            file.delete();
        }
        File dir = new File("/myself/tmp/dl4j/gan/data/train/0");
        if (dir.exists()) {
            dir.delete();
        }

        File file1 = new File("/myself/mnist/mnist_png/training/0/1.png");
        if (file1.exists()) {
            BufferedImage image = ImageIO.read(file1);
            realPic = ImageUtils.load(file1.getAbsolutePath()).div(255);
            width = image.getWidth();
            height = image.getHeight();
        }

        discInit();
        discTrueTrain(1);
        initGen();


        int n = 10000;

        /**
        while (n-- > 0) {
            genTrain(1);
            discFlaseTrain(1);
        }
         */
        while (n-- > 0) {
            discTrueTrain(1);

            genTrain(1);

            discFlaseTrain(1);

            /*while (true) {
                double[] discTrueRet = discTrueTrain(1);
                double cha = discTrueRet[1] - discTrueRet[0];
                if (cha > 0.05) {
                    break;
                }
            }
            while (true) {
                double[] genRet = genTrain(1);
                double cha = genRet[1] - genRet[0];
                if (cha > 0.05 || genRet[0] > 0.7) {
                    break;
                }
            }
            while (true) {
                double[] discFalseRet = discFlaseTrain(1);
                double cha = Math.abs(discFalseRet[0] - discFalseRet[1]);
                if (cha < 0.05 || discFalseRet[0] > 0.55) {
                    break;
                }
            }*/


        }

    }

    private static INDArray pic;
    private static INDArray getRealPic() {
        if (pic == null) {
            Random rd = new Random();
            int dd = 6;//rd.nextInt(10);
            String dir = "/myself/mnist/mnist_png/training/" + dd;
            File dirFile = new File(dir);
            File[] files = dirFile.listFiles();
            String path = files[rd.nextInt(files.length)].getAbsolutePath();
            //String path = files[0].getAbsolutePath();
            pic = ImageUtils.load(path);
        }
        return pic;
    }

    private static int startIdx = 0;
    private static INDArray input;
    private static INDArray getNorseInputData() {
        Random rd = new Random();
        return Nd4j.randn(new int[]{1, 10}, 1234);
    }

    private static INDArray label1;
    private static INDArray getLabelData1() {
        if (label1 == null) {
            label1 = Nd4j.create
                    (new double[]{0.0, 1.0},
                            new int[]{1, 2});
        }
        return label1;
    }
/*
    private static INDArray label11;
    private static INDArray getLabelData11() {
        if (label11 == null) {
            label11 = Nd4j.create
                    (new double[]{0.0, 7},
                            new int[]{1, 2});
        }
        return label11;
    }*/

    private static INDArray label0;
    private static INDArray getLabelData0() {
        if (label0 == null) {
            label0 = Nd4j.create
                    (new double[]{1.0, 0.0},
                            new int[]{1, 2});
        }
        return label0;
    }

    private static boolean eq(INDArray a1, INDArray a2) {
        Number number = a1.eq(a2).minNumber();
        return number.intValue() == 1;
    }

    private static Map<String, INDArray> copy(Map<String, INDArray> data) {
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

    private static boolean eq(Map<String, INDArray> param1, Map<String, INDArray> param2, int start, boolean m) {
        boolean eq = true;
        Set<String> genKeySet = param1.keySet();
        for (Iterator<String> iter = genKeySet.iterator(); iter.hasNext();) {
            String key = iter.next();
            String[] keys = key.split("_");
            Integer keyIdx = Integer.parseInt(keys[0]);
            if (keyIdx >= start) {
                int idx = keyIdx - start;
                if (!m) {
                    idx = keyIdx;
                }
                INDArray disc = param2.get(idx + "_" + keys[1]);

                INDArray gen = param1.get(key);
                if (!eq(disc, gen)) {
                    eq = false;
                }
            }
        }
        return eq;
    }

    private static boolean eq1(Map<String, INDArray> param1, Map<String, INDArray> param2) {
        boolean eq = true;
        Set<String> genKeySet = param1.keySet();
        for (Iterator<String> iter = genKeySet.iterator(); iter.hasNext();) {
            String key = iter.next();
            INDArray disc = param2.get(key);

            INDArray gen = param1.get(key);
            if (!eq(disc, gen)) {
                eq = false;
            }
        }
        return eq;
    }

    private static int anlyLayerCha(Set<String> keySet1, Set<String> keySet2) {
        int max1 = maxIdx(keySet1);
        int max2 = maxIdx(keySet2);
        return Math.abs(max1 - max2);
    }

    private static int maxIdx(Set<String> keySet) {
        String[] keyArr = new String[keySet.size()];
        keyArr = keySet.toArray(keyArr);

        int max = -1;
        for (int i = 0; i < keyArr.length; i++) {
            Integer idx = Integer.parseInt(keyArr[i].split("_")[0]);

            if (idx > max) {
                max = idx;
            }
        }
        return max;
    }

    private static void initGen() {
        if (genNet == null) {
            MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                    .seed(1234)
                    .weightInit(WeightInit.XAVIER)
                    .list()
                    .layer(new DenseLayer.Builder()
                            .nIn(10)
                            .nOut(1024)
                            .activation(Activation.RELU)
                            .updater(new RmsProp(grate))
                            .build())
                    .layer(new BatchNormalization.Builder()
                            .decay(1)
                            .updater(new RmsProp(grate))
                            .build())
                    .layer(new DenseLayer.Builder()
                            .nOut(512)
                            .activation(Activation.RELU)
                            .updater(new RmsProp(grate))
                            .build())
                    .layer(new BatchNormalization.Builder()
                            .decay(1)
                            .updater(new RmsProp(grate))
                            .build())
                    .layer(new DenseLayer.Builder()
                            .nOut(width * height)
                            .activation(Activation.SIGMOID)
                            .updater(new RmsProp(grate))
                            .build())







                    .layer(new DenseLayer.Builder()
                            .nIn(width * height)
                            .nOut(1024)
                            .activation(Activation.RELU)
                            .updater(new RmsProp(0))
                            .build())
                    .layer(new DenseLayer.Builder()
                            .nIn(1024)
                            .nOut(512)
                            .activation(Activation.RELU)
                            .updater(new RmsProp(0))
                            .build())
                    .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                            .nIn(512)
                            .nOut(2)
                            .activation(Activation.SOFTMAX)
                            .updater(new RmsProp(0))
                            .build())
                    .backprop(true).pretrain(false).build();

            genNet = new MultiLayerNetwork(conf);
            genNet.init();
            genNet.setListeners(new ScoreIterationListener(10));

            startIdx = anlyLayerCha(genNet.paramTable().keySet(), discNet.paramTable().keySet());
            conf.getInputPreProcessors().put(startIdx, new InOutputPlatPreProcessor(width, height, channel));
        }
    }

    private static void discCp2GenParam() {
        Map<String, INDArray> paramTableGen = genNet.paramTable();

        //恢复数据
        Set<String> genKeySet = paramTableGen.keySet();
        for (Iterator<String> iter = genKeySet.iterator(); iter.hasNext();) {
            String key = iter.next();
            String[] keys = key.split("_");
            Integer keyIdx = Integer.parseInt(keys[0]);
            if (keyIdx >= startIdx) {
                String discKey = (keyIdx - startIdx) + "_" + keys[1];
                INDArray item = discNet.getParam(discKey);
                Nd4j.copy(item, genNet.getParam(key));
            }
        }
    }


    private static MultiLayerNetwork genNet;
    private static double[] genTrain(int count) {
        discCp2GenParam();

        INDArray input = getNorseInputData();
        INDArray label = getLabelData1();


        System.out.println();
        double[] retArr = null;
        for (int i = 0; i < count; i++) {
            Map<String, INDArray> genParamCopy = copy(genNet.paramTable());
            genNet.fit(input, label);
            System.out.println("生成模型-前后参数变化情况=" + eq1(genParamCopy, genNet.paramTable()));
            InOutputPlatPreProcessor.gen = true;
            INDArray retLabel1 = genNet.output(input);
            InOutputPlatPreProcessor.gen = false;
            System.out.println((i + 1) + "_生成模型-假当真(训练后)=" + retLabel1);
            retArr = retLabel1.toDoubleVector();
        }
//        System.out.println("1_生成模型-前后评估模型参数对比=" + eq(genNet.paramTable(), discNet.paramTable(), 14, true));
        System.out.println();
        /**
        for (int i = 0; i < genKeyList.size(); i++) {
            System.out.println(genKeyList.get(i) + "_" + discKeyList.get(i) + "_eq="
                    + eq(genNet.getParam(genKeyList.get(i)), discNet.getParam(discKeyList.get(i))));
        }
         */


        return retArr;
    }

    private static MultiLayerNetwork discNet;
    private static RmsProp discUpdater;
    private static void discInit() throws IOException {
        discUpdater = new RmsProp(drate);
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(1234)
                .updater(new RmsProp(1e-3))
                .weightInit(WeightInit.XAVIER)
                .list()
                .layer(new DenseLayer.Builder()
                        .nIn(width * height)
                        .nOut(1024)
                        .activation(Activation.RELU)
                        .build())
                .layer(new DenseLayer.Builder()
                        .nIn(1024)
                        .nOut(512)
                        .activation(Activation.RELU)
                        .build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nIn(512)
                        .nOut(2)
                        .activation(Activation.SOFTMAX)
                        .build())
                .setInputType(InputType.feedForward(width * height))
                .backprop(true).pretrain(false).build();

        discNet = new MultiLayerNetwork(conf);
        discNet.init();
        discNet.setListeners(new ScoreIterationListener(10));

    }


    public static int[] trimRGBColor(int color) {
        int[] rgb = new int[4];
        rgb[0] = (color >>24) & 0xff;
        rgb[1] = (color >> 16) & 0xff;
        rgb[2] = (color >> 8) & 0xff;
        rgb[3] = color & 0x000000ff;

        return rgb;
    }

    private static double[] temp2Arr;
    private static double[] toDbArr(INDArray data) {
        long[] shape = data.shape();
        int width = (int) shape[3];
        int heith = (int) shape[2];

        double[] dataDb = data.permute(0, 3, 2, 1).getRow(0).data().asDouble();
        return dataDb;
    }

    private static INDArray toINDArray(double[] dataDb) {
        int[] shape = new int[] {channel, height, width};

        INDArray ret2 = Nd4j.create(1, width * height * channel);
        int len = width * height;
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int idx = y * width + x;
                ret2.putScalar(idx, dataDb[y * width + x]);
                if (channel > 1) {
                    ret2.putScalar(idx + len, dataDb[y * width + x + len]);
                }
                if (channel > 2) {
                    ret2.putScalar(idx + len * 2, dataDb[y * width + x + len * 2]);
                }
            }
        }
        return Nd4j.expandDims(ret2.reshape(shape), 0);
    }

    private static int[] tempArr;
    private static int[] toColorArr(INDArray data) {
        long[] shape = data.shape();
        int width = (int) shape[3];
        int heith = (int) shape[2];

        int[] dataPixels = data.permute(0, 3, 2, 1).getRow(0).data().asInt();

        int[] pixels = new int[width * heith];

        for (int y = 0; y < heith; y++) {
            for (int x = 0; x < width; x++) {
                int[] rgb = new int[4];
                rgb[0] = 0xff;
                rgb[1] = dataPixels[y * width + x];
                rgb[2] = dataPixels[y * width + x + pixels.length];
                rgb[3] = dataPixels[y * width + x + pixels.length * 2];
                pixels[y * width + x] = rgb[0] << 24 | rgb[1] << 16 | rgb[2] << 8 | rgb[3];
            }
        }
        return pixels;
    }

    private static INDArray toINDArray(int[] colorArr) {
        int[] shape = new int[] {channel, height, width};

        INDArray ret2 = Nd4j.create(1, width * height * channel);

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int idx = y * width + x;
                int color = colorArr[idx];
                int[] argb = trimRGBColor(color);
                ret2.putScalar(idx, (argb[1]) & 0xFF);
                ret2.putScalar(idx + colorArr.length, (argb[2]) & 0xFF);
                ret2.putScalar(idx + colorArr.length * 2, (argb[3]) & 0xFF);
            }
        }
        return Nd4j.expandDims(ret2.reshape(shape), 0);
    }

    private static BufferedImage toBufferedImage(INDArray data) {
        long[] shape = data.shape();
        int width = (int) shape[3];
        int heith = (int) shape[2];
        int chl = (int) shape[1];

        BufferedImage image = new BufferedImage(width, heith, BufferedImage.TYPE_3BYTE_BGR);

        int[] dataPixels = data.permute(0, 3, 2, 1).getRow(0).data().asInt();

        int[] pixels = new int[width * heith];

        for (int y = 0; y < heith; y++) {
            for (int x = 0; x < width; x++) {
                int[] rgb = new int[4];
                rgb[0] = 0xff;
                rgb[1] = dataPixels[y * width + x];
                rgb[2] = dataPixels[y * width + x + pixels.length];
                rgb[3] = dataPixels[y * width + x + pixels.length * 2];
                pixels[y * width + x] = rgb[0] << 24 | rgb[1] << 16 | rgb[2] << 8 | rgb[3];
            }
        }
        setRGB(image, 0, 0, width, heith, pixels);
        return image;
    }

    public static int[] getRGB( BufferedImage image, int x, int y, int width, int height, int[] pixels ) {
        int type = image.getType();
        if ( type == BufferedImage.TYPE_INT_ARGB || type == BufferedImage.TYPE_INT_RGB )
            return (int [])image.getRaster().getDataElements( x, y, width, height, pixels );
        return image.getRGB( x, y, width, height, pixels, 0, width );
    }

    public static void setRGB( BufferedImage image, int x, int y, int width, int height, int[] pixels ) {
        int type = image.getType();
        if ( type == BufferedImage.TYPE_INT_ARGB || type == BufferedImage.TYPE_INT_RGB )
            image.getRaster().setDataElements( x, y, width, height, pixels );
        else
            image.setRGB( x, y, width, height, pixels, 0, width );
    }

    private static INDArray realPic = null;
    private static double[] discFlaseTrain(int count) throws IOException {
        Double old = discUpdater.getLearningRate();
        discUpdater.setLearningRate(dfrate);
        double[] retArr = null;
        System.out.println();
        try {
            for (int i = 0; i < count; i++) {
                if (InOutputPlatPreProcessor.temp2Arr != null) {
                    INDArray indArray0 = toINDArray(InOutputPlatPreProcessor.temp2Arr);
                    indArray0 = indArray0.reshape(1, width * height);
                    INDArray label = getLabelData0();
                    Map<String, INDArray> discParamCopy = copy(discNet.paramTable());
                    discNet.fit(indArray0, label);
                    //System.out.println("评估模型-假图-前后参数变化情况=" + eq1(discParamCopy, discNet.paramTable()));
                    INDArray retLabel1 = discNet.output(indArray0);
                    System.out.println((i + 1) + "_评估模型-真当假(训练后)=" + retLabel1);
                    retArr = retLabel1.toDoubleVector();
                }
            }
        } finally {
            discUpdater.setLearningRate(old);
        }


        System.out.println();
        return retArr;
    }

    private static double[] discTrueTrain(int count) throws IOException {
        double[] retArr = null;
        System.out.println();
        for (int i = 0; i < count; i++) {
            INDArray label = getLabelData1();
            INDArray input = getRealPic().reshape(1, width * height).div(255);
            INDArray retLabel0 = discNet.output(input);
            System.out.println((i + 1) + "_评估模型-真当真(训练前)=" + retLabel0);
            Map<String, INDArray> discParamCopy = copy(discNet.paramTable());
            discNet.fit(input, label);
            //System.out.println("评估模型-真图-前后参数变化情况=" + eq1(discParamCopy, discNet.paramTable()));
            INDArray retLabel1 = discNet.output(input);
            System.out.println((i + 1) + "_评估模型-真当真(训练后)=" + retLabel1);
            retArr = retLabel1.toDoubleVector();
        }
        System.out.println();
        return retArr;
    }
}
