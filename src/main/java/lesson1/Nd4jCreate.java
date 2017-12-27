package lesson1;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Created by Joe on 2017/12/27.
 * Nd4j创建Nd4j的一些基本操作
 */
public class Nd4jCreate {
    public static void main(String[] args) {
        /*
            构造一个3行5列的全0  ndarray
         */
        INDArray zeros = Nd4j.zeros(3, 5);
        System.out.println(zeros);

        /*
            构造一个3行5列的全1 ndarray
         */
        INDArray ones = Nd4j.ones(3, 5);
        System.out.println(ones);

        /*
            构造一个3行5列，数组元素均为随机产生的ndarray
         */
        INDArray rands = Nd4j.rand(3, 5);
        System.out.println(rands);


        /*
            构造一个3行5列，数组元素服从高斯分布（平均值为0，）
         */
        INDArray randns = Nd4j.randn(3, 5);
        System.out.println(randns);
    }
}
