package lesson7;

import java.io.File;

/**
 * Created by Joe on 2018/6/22.
 */
public class GenerateData {
    public static void main(String[] args) {
        String baseDirPath = "data/";
        String[] fileNames = {"train.csv", "test.csv"};
        int[] dataNums = {3000, 1000};

        File baseDir = new File(baseDirPath);
        boolean mkDir = baseDir.exists() || baseDir.mkdirs();
    }
}
