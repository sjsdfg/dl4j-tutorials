package lesson1;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Created by Joe on 2017/12/27.
 * 获取并且设置数组部分
 */
public class Nd4jGetAndSetParts {
    public static void main(String[] args) {
        INDArray nd = Nd4j.create(new float[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, new int[]{2, 6});
        System.out.println("原始数组");
        System.out.println(nd);

        /*
            获取一行
         */
        System.out.println("获取数组中的一行");
        INDArray singleRow = nd.getRow(0);
        System.out.println(singleRow);

        /*
            获取多行
         */
        System.out.println("获取数组中的多行");
        INDArray multiRows = nd.getRows(0, 1);
        System.out.println(multiRows);

        /*
            替换其中的一行
         */
        System.out.println("替换原有数组中的一行");
        INDArray replaceRow = Nd4j.create(new float[]{1, 3, 5, 7, 9, 11});
        nd.putRow(0, replaceRow);
        System.out.println(nd);
    }
}
