package lesson7;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

/**
 * Created by Joe on 2018/6/22.
 */
public class GenerateData {
    private static double step = 0.01;

    public static void main(String[] args) {
        // 根目录
        String baseDirPath = "data/";
        // 文件名称
        String[] fileNames = {"train.csv", "test.csv"};
        // 构造的数据条数
        int[] dataNums = {3000, 1000};

        File baseDir = new File(baseDirPath);
        boolean mkDir = baseDir.exists() || baseDir.mkdirs();


    }

    /**
     * 构造数据并且写入文件
     * @param baseDir 根目录
     * @param fileName 文件名
     * @param start 循环开始位置
     * @param nums 数据条数
     */
    public void writeFile(File baseDir, String fileName, int start, int nums) {
        // 接下来创造训练集数据
        File train = new File(baseDir, fileName);
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(train))) {
            for (int i = start; i < nums; i++) {
                double x = Math.sin(i * step);
                double y = Math.cos(i * step);
                writer.write(x + "," + y);
                writer.newLine();
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
