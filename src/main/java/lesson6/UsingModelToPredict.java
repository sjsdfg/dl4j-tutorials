package lesson6;

import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.io.ClassPathResource;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.Arrays;

/**
 * Created by Joe on 2018/6/14.
 */
public class UsingModelToPredict {
    private static int height = 28;
    private static int width = 28;
    private static int channel = 1;

    private static String modelPath = "model/LeNetModel.zip";

    public static void main(String[] args) throws IOException {
        /**
         * 首先我们需要创建一个数据读取器
         *
         * channel为1的时候，如果我们输入的一个彩色图，会自动的进行图像的灰度处理
         */
        NativeImageLoader imageLoader = new NativeImageLoader(height, width, channel);

        /**
         * 指定我们的图片位置
         */
        File imgFile = new ClassPathResource("/mnist/4.jpg").getFile();

        /**
         * 使用ImageLoader把图像转化为 INDArray
         */
        INDArray imgNdarray = imageLoader.asMatrix(imgFile);

//        DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
//        scaler.transform(imgNdarray);

//        imgNdarray = imgNdarray.reshape(1, channel * height * width);

        /**
         * 打印查看数据
         */
//        System.out.println(imgNdarray);

        /**
         * 打印查看我们的数据的shape
         * 我们数据的相撞
         */
        System.out.println(Arrays.toString(imgNdarray.shape()));

        /**
         * 把灰度化之后的图片进行保存
         */
        ImageIO.write(imageFromINDArray(imgNdarray), "jpg", new File("model/test.jpg"));


        /**
         * 反序列化我们的模型
         * LeNet -> 卷积神经网络 -> 输入数据要求为4个维度 -> [batch, channel, height, width]
         * SingleLayer -> 全连接神经网络 -> 输入的数据要求的维度为2 -> [batch, features] -> [batch, channel * height * width]
         */
        MultiLayerNetwork network = ModelSerializer.restoreMultiLayerNetwork(modelPath, false);

        /**
         * 分类使用one-hot编码
         * 输出为概率
         * 概率最大的位置，为我们所分类的类型
         */
        INDArray output = network.output(imgNdarray);

        System.out.println(output);
        /**
         * 获取最大值的索引
         */
        System.out.println(Nd4j.getBlasWrapper().iamax(output));

        /**
         * output->只能输出模型的最后一层的经过激活函数的值 -> softmax
         * predict -> 其实就是一个 output 的封装 -> 只能用于分类
         */
        int[] results = network.predict(imgNdarray);
        System.out.println(Arrays.toString(results));

//        System.out.println(imageLoader.asMatrix(new File("model/test.jpg")));
    }


    /**
     * 将单通道的 INDArray 保存为灰度图
     * @param array 输入
     * @return 灰度图转化
     */
    private static BufferedImage imageFromINDArray(INDArray array) {
        int[] shape = array.shape();

        int height = shape[2];
        int width = shape[3];
        BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY);
        for (int x = 0; x < width; x++) {
            for (int y = 0; y < height; y++) {
                int gray = array.getInt(0, 0, y, x);

                // handle out of bounds pixel values
                gray = Math.min(gray, 255);
                gray = Math.max(gray, 0);

                image.getRaster().setSample(x, y, 0, gray);
            }
        }
        return image;
    }
}
