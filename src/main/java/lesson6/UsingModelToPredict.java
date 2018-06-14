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
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.lang.annotation.Native;
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
        NativeImageLoader imageLoader = new NativeImageLoader(height, width, channel);

        File imgFile = new ClassPathResource("/mnist/4.jpg").getFile();

        INDArray imgNdarray = imageLoader.asMatrix(imgFile);

//        DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
//        scaler.transform(imgNdarray);
        System.out.println(imgNdarray);

        System.out.println(Arrays.toString(imgNdarray.shape()));

        ImageIO.write(imageFromINDArray(imgNdarray), "jpg", new File("model/test.jpg"));


        MultiLayerNetwork network = ModelSerializer.restoreMultiLayerNetwork(modelPath, false);

        INDArray output = network.output(imgNdarray);
//
//        System.out.println(output);
//        System.out.println(Nd4j.getBlasWrapper().iamax(output));
//
//        int[] results = network.predict(imgNdarray);
//        System.out.println(Arrays.toString(results));

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
