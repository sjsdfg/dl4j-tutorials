package baidudianshi;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * Created by Joe on 2018/10/18.
 * 数据来源
 * http://dianshi.baidu.com/dianshi/pc/competition/22/rule
 * “探寻地球密码”天宫数据利用大赛
 * 百度点石初赛数据：https://pan.baidu.com/s/1_M0yPejFTvxDFOn4780OPA
 */
public class PrepareData {

    public static void main(String[] args) {

        List<Category> categories = new ArrayList<>();

        // 使用 tryWithResource 进行文件的读取
        try (BufferedReader reader = new BufferedReader(new FileReader("data/宽波段数据集-预赛训练集2000/光学数据集-预赛训练集-2000-有标签.csv"))) {
            String tmp;
            while ((tmp = reader.readLine()) != null) {
                String[] strs = tmp.split(",");
                categories.add(new Category(strs[1], strs[0]));
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        // 采用 java 8 的 stream 对数据类别进行分类
        Map<String, List<Category>> categoryMap = categories.stream()
                .collect(Collectors.groupingBy(Category::getLabelName));

        // 遍历分类结果以及每一类的数据个数
        categoryMap.forEach((key, value) -> System.out.println(key + ":" + value.size()));

        // 根据分类结果在 data 文件夹下面构造数据目录
        final String categoryDirPrefix = "data/train/";
        final String sourceDirPrefix = "data/宽波段数据集-预赛训练集2000/预赛训练集-2000/";
        categoryMap.forEach((key, value) -> {
            // 构造目录
            File file = new File(categoryDirPrefix + key);
            if (!file.exists()) file.mkdir();

            // 然后遍历value，把标签文件转移到对应的文件夹下
            value.forEach(category -> {
                Path source = Paths.get(sourceDirPrefix + category.pictureName);
                Path target = Paths.get(file.getAbsolutePath() + "/" + category.pictureName);

                System.out.println(source + ":" + target);

                try {
                    // Path copy(Path source, Path target, CopyOption... options)
                    Files.copy(source, target);
                } catch (IOException e) {
                }
            });
        });
    }

    static class Category {
        private String labelName;
        private String pictureName;

        public Category(String labelName, String pictureName) {
            this.labelName = labelName;
            this.pictureName = pictureName;
        }

        public String getLabelName() {
            return labelName;
        }

        public void setLabelName(String labelName) {
            this.labelName = labelName;
        }

        public String getPictureName() {
            return pictureName;
        }

        public void setPictureName(String pictureName) {
            this.pictureName = pictureName;
        }

        @Override
        public String toString() {
            return "Category{" +
                    "labelName='" + labelName + '\'' +
                    ", pictureName='" + pictureName + '\'' +
                    '}';
        }
    }
}
