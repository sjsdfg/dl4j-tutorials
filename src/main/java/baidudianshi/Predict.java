package baidudianshi;

import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.Normalizer;
import org.nd4j.linalg.primitives.Pair;

import javax.naming.Name;
import java.io.File;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Created by Joe on 2018/10/18.
 */
public class Predict {
    public static void main(String[] args) throws Exception {
        String testPath = "data/test";
        File testDir = new File(testPath);
        File[] files = testDir.listFiles();

        Pair<MultiLayerNetwork, Normalizer> modelAndNormalizer = ModelSerializer
                .restoreMultiLayerNetworkAndNormalizer(new File("model/AlexNet.zip"), false);

        NativeImageLoader imageLoader = new NativeImageLoader(256, 256, 3);

        MultiLayerNetwork network = modelAndNormalizer.getFirst();
        DataNormalization normalizer = (DataNormalization) modelAndNormalizer.getSecond();

        Map<Integer, String> map = new HashMap<>();
        map.put(0, "CITY");
        map.put(1, "DESERT");
        map.put(2, "FARMLAND");
        map.put(3, "LAKE");
        map.put(4, "MOUNTAIN");
        map.put(5, "OCEAN");

        for (File file : files) {
            INDArray indArray = imageLoader.asMatrix(file);
            normalizer.transform(indArray);

            int[] values = network.predict(indArray);
            String label = map.get(values[0]);

            System.out.println(file.getName() + "," + label);
        }
    }
}
